import torch
from omni_speech.model.builder import load_pretrained_model
from omni_speech.datasets.preprocess import preprocess_llama_3_v1, preprocess_qwen_2_5_v1
import whisper
import argparse
import time
import torch.nn.functional as F
import os
import warnings
from contextlib import contextmanager

# 성능 최적화를 위한 환경 설정
def setup_performance_env():
    """성능 최적화를 위한 환경 변수 설정"""
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # 메모리 할당 최적화
    if hasattr(torch.cuda, 'memory_pool'):
        torch.cuda.empty_cache()

@contextmanager
def optimize_cuda_context():
    """CUDA 컨텍스트 최적화"""
    try:
        # CUDA 그래프 캐싱 활성화
        with torch.cuda.stream(torch.cuda.Stream()):
            yield
    finally:
        torch.cuda.synchronize()

# 환경 설정 실행
setup_performance_env()

VOCALNET_MODEL = "checkpoints/Llama-3.2-1B-Instruct-s2t"    #  /root/speechllm_checkpoints/VocalNet-qwen25-7B  VocalNet speech LLM   i.e. ./checkpoints/VocalNet-1B

class VocalNetModel:
    def __init__(self, model_name_or_path: str, **kwargs):
        self.model_name_or_path = model_name_or_path
        self.empty = True

        self.temperature = kwargs.get('temperature', 0)
        self.num_beams = kwargs.get('num_beams', 1)
        self.max_new_tokens = kwargs.get('max_new_tokens', 512)
        self.top_p = kwargs.get('top_p', 0.1)
        
        # 성능 최적화 옵션 추가
        self.use_compile = kwargs.get('use_compile', True)
        self.use_flash_attention = kwargs.get('use_flash_attention', True)
        self.optimize_memory = kwargs.get('optimize_memory', True)
        self.use_mixed_precision = kwargs.get('use_mixed_precision', True)
        self.use_kv_cache_optimization = kwargs.get('use_kv_cache_optimization', True)

        self.empty = True

    def __initilize__(self):
        if self.empty:
            self.empty = False
            
            # GPU 메모리 최적화
            if self.optimize_memory:
                torch.cuda.empty_cache()
                
            # Mixed Precision 설정
            if self.use_mixed_precision:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
            self.tokenizer, self.model, _ = load_pretrained_model(self.model_name_or_path, s2s=False)
            
            # 모델을 evaluation 모드로 설정
            self.model.eval()
            
            # KV-Cache 최적화
            if self.use_kv_cache_optimization and hasattr(self.model, 'config'):
                if hasattr(self.model.config, 'use_cache'):
                    self.model.config.use_cache = True
                # Attention 최적화
                if hasattr(self.model.config, 'attention_dropout'):
                    self.model.config.attention_dropout = 0.0
            
            # Flash Attention 활성화 (가능한 경우)
            if self.use_flash_attention and hasattr(self.model.config, 'use_flash_attention_2'):
                self.model.config.use_flash_attention_2 = True
            
            # 모델 컴파일 (PyTorch 2.0+)
            if self.use_compile and hasattr(torch, 'compile'):
                print("모델을 컴파일 중입니다... (첫 실행 시 시간이 걸릴 수 있습니다)")
                try:
                    # 다양한 컴파일 모드 시도
                    compile_mode = "reduce-overhead"  # 또는 "max-autotune", "default"
                    self.model = torch.compile(self.model, mode=compile_mode, dynamic=False)
                    print(f"모델 컴파일 완료 (모드: {compile_mode})")
                except Exception as e:
                    print(f"모델 컴파일 실패: {e}")
                    self.use_compile = False
            
            # GPU 워밍업
            self._warmup_model()

    def _warmup_model(self):
        """모델 워밍업을 통한 초기 오버헤드 제거"""
        print("모델 워밍업 중...")
        try:
            with optimize_cuda_context():
                dummy_input_ids = torch.randint(0, 1000, (1, 50), device='cuda')
                dummy_speech = torch.randn(1, 3000, 128, device='cuda', dtype=torch.float16)
                dummy_speech_length = torch.LongTensor([3000]).to('cuda')
                
                # Mixed Precision 워밍업
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        with torch.inference_mode():
                            _ = self.model.generate(
                                dummy_input_ids,
                                speech=dummy_speech,
                                speech_lengths=dummy_speech_length,
                                do_sample=False,
                                max_new_tokens=5,
                                use_cache=True,
                                pad_token_id=128004,
                            )
                else:
                    with torch.inference_mode():
                        _ = self.model.generate(
                            dummy_input_ids,
                            speech=dummy_speech,
                            speech_lengths=dummy_speech_length,
                            do_sample=False,
                            max_new_tokens=5,
                            use_cache=True,
                            pad_token_id=128004,
                        )
                
            print("워밍업 완료!")
        except Exception as e:
            print(f"워밍업 중 오류 발생 (무시됨): {e}")

    def __call__(self, messages: list) -> str:
        """
        "infer_messages": [[{'role': 'user', 'content': '<speech>', 'path': ./OpenAudioBench/eval_datas/alpaca_eval/audios/alpaca_eval_198.mp3'}]
        """
        start_time = time.time()
        
        audio_path = messages[0]['path']
        speech = whisper.load_audio(audio_path)
        
        # 오디오 전처리 최적화
        if self.model.config.speech_encoder_type == "glm4voice":
            speech_length = torch.LongTensor([speech.shape[0]])
            speech = torch.from_numpy(speech)
            speech = F.layer_norm(speech, speech.shape)  # torch.nn.functional 대신 F 사용
        else:
            raw_len = len(speech)
            speech = whisper.pad_or_trim(speech)
            padding_len = len(speech)
            speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0).unsqueeze(0)
            speech_length = round(raw_len / padding_len * 3000 + 0.5)
            speech_length = torch.LongTensor([speech_length])
        
        conversation = [{"from": "human", "value": "<speech>", "path": f"{audio_path}"}]
        
        # 토크나이저 처리 최적화
        if 'qwen' in self.model_name_or_path.lower():
            input_ids = preprocess_qwen_2_5_v1([conversation], self.tokenizer, True, 4096)['input_ids']
            input_ids = torch.cat([input_ids.squeeze(), torch.tensor([198, 151644, 77091, 198], device=input_ids.device)]).unsqueeze(0)
        else:
            input_ids = preprocess_llama_3_v1([conversation], self.tokenizer, True, 4096)['input_ids']
            input_ids = torch.cat([input_ids.squeeze(), torch.tensor([128006, 78191, 128007, 271], device=input_ids.device)]).unsqueeze(0)

        # GPU로 전송 최적화 (non_blocking=True)
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        speech_tensor = speech.to(dtype=torch.float16, device='cuda', non_blocking=True)
        speech_length = speech_length.to(device='cuda', non_blocking=True)
        
        # 추론 실행 with Mixed Precision
        with optimize_cuda_context():
            # CUDA 스트림 동기화
            torch.cuda.synchronize()
            inference_start = time.time()
            
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    with torch.inference_mode():
                        outputs = self.model.generate(
                            input_ids,
                            speech=speech_tensor,
                            speech_lengths=speech_length,
                            do_sample=True if self.temperature > 0 else False,
                            temperature=self.temperature,
                            top_p=self.top_p if self.top_p is not None else 0.0,
                            num_beams=self.num_beams,
                            max_new_tokens=self.max_new_tokens,
                            use_cache=True,
                            pad_token_id=128004,
                        )
            else:
                with torch.inference_mode():
                    outputs = self.model.generate(
                        input_ids,
                        speech=speech_tensor,
                        speech_lengths=speech_length,
                        do_sample=True if self.temperature > 0 else False,
                        temperature=self.temperature,
                        top_p=self.top_p if self.top_p is not None else 0.0,
                        num_beams=self.num_beams,
                        max_new_tokens=self.max_new_tokens,
                        use_cache=True,
                        pad_token_id=128004,
                    )
            
            torch.cuda.synchronize()
            inference_end = time.time()
            
            output_ids = outputs
            output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        total_time = time.time() - start_time
        inference_time = inference_end - inference_start
        
        print(f"총 처리 시간: {total_time:.3f}초, 순수 추론 시간: {inference_time:.3f}초")
            
        result = {"text": output_text.strip(), "inference_time": inference_time, "total_time": total_time}
        return result

    def batch_call(self, batch_messages: list) -> list:
        """
        배치 처리를 통한 다중 오디오 파일 동시 처리
        batch_messages: [[{'role': 'user', 'content': '<speech>', 'path': 'audio1.mp3'}], 
                        [{'role': 'user', 'content': '<speech>', 'path': 'audio2.mp3'}], ...]
        """
        if not batch_messages:
            return []
            
        start_time = time.time()
        batch_size = len(batch_messages)
        
        # 배치 오디오 전처리
        speech_tensors = []
        speech_lengths = []
        input_ids_batch = []
        
        for messages in batch_messages:
            audio_path = messages[0]['path']
            speech = whisper.load_audio(audio_path)
            
            if self.model.config.speech_encoder_type == "glm4voice":
                speech_length = torch.LongTensor([speech.shape[0]])
                speech = torch.from_numpy(speech)
                speech = F.layer_norm(speech, speech.shape)
            else:
                raw_len = len(speech)
                speech = whisper.pad_or_trim(speech)
                padding_len = len(speech)
                speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0).unsqueeze(0)
                speech_length = round(raw_len / padding_len * 3000 + 0.5)
                speech_length = torch.LongTensor([speech_length])
            
            speech_tensors.append(speech)
            speech_lengths.append(speech_length)
            
            # 토크나이저 처리
            conversation = [{"from": "human", "value": "<speech>", "path": f"{audio_path}"}]
            
            if 'qwen' in self.model_name_or_path.lower():
                input_ids = preprocess_qwen_2_5_v1([conversation], self.tokenizer, True, 4096)['input_ids']
                input_ids = torch.cat([input_ids.squeeze(), torch.tensor([198, 151644, 77091, 198], device='cuda')]).unsqueeze(0)
            else:
                input_ids = preprocess_llama_3_v1([conversation], self.tokenizer, True, 4096)['input_ids']
                input_ids = torch.cat([input_ids.squeeze(), torch.tensor([128006, 78191, 128007, 271], device='cuda')]).unsqueeze(0)
            
            input_ids_batch.append(input_ids)
        
        # 배치 텐서 생성
        max_seq_len = max(ids.shape[1] for ids in input_ids_batch)
        padded_input_ids = []
        
        for input_ids in input_ids_batch:
            if input_ids.shape[1] < max_seq_len:
                padding = torch.full((1, max_seq_len - input_ids.shape[1]), 
                                   self.tokenizer.pad_token_id or 0, 
                                   device=input_ids.device, dtype=input_ids.dtype)
                input_ids = torch.cat([input_ids, padding], dim=1)
            padded_input_ids.append(input_ids)
        
        batch_input_ids = torch.cat(padded_input_ids, dim=0)
        batch_speech = torch.cat(speech_tensors, dim=0)
        batch_speech_lengths = torch.cat(speech_lengths, dim=0)
        
        # GPU로 전송
        batch_input_ids = batch_input_ids.to(device='cuda', non_blocking=True)
        batch_speech = batch_speech.to(dtype=torch.float16, device='cuda', non_blocking=True)
        batch_speech_lengths = batch_speech_lengths.to(device='cuda', non_blocking=True)
        
        # 배치 추론 실행
        with torch.inference_mode():
            torch.cuda.synchronize()
            inference_start = time.time()
            
            outputs = self.model.generate(
                batch_input_ids,
                speech=batch_speech,
                speech_lengths=batch_speech_lengths,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p if self.top_p is not None else 0.0,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id or 128004,
            )
            
            torch.cuda.synchronize()
            inference_end = time.time()
            
            # 배치 디코딩
            output_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        total_time = time.time() - start_time
        inference_time = inference_end - inference_start
        
        print(f"배치 크기: {batch_size}, 총 처리 시간: {total_time:.3f}초, "
              f"순수 추론 시간: {inference_time:.3f}초, "
              f"파일당 평균 시간: {total_time/batch_size:.3f}초")
        
        results = []
        for i, text in enumerate(output_texts):
            results.append({
                "text": text.strip(),
                "inference_time": inference_time / batch_size,
                "total_time": total_time / batch_size,
                "batch_size": batch_size
            })
        
        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VocalNet infer')
    parser.add_argument('--query_audio', type=str, default="./omni_speech/infer/llama_questions_42.wav")
    parser.add_argument('--save_dir', default="./generated_audio", required=False)
    
    # 성능 최적화 옵션 추가
    parser.add_argument('--use_compile', action='store_true', default=True, help='PyTorch 2.0 모델 컴파일 사용')
    parser.add_argument('--no_compile', dest='use_compile', action='store_false', help='모델 컴파일 비활성화')
    parser.add_argument('--use_flash_attention', action='store_true', default=True, help='Flash Attention 사용')
    parser.add_argument('--optimize_memory', action='store_true', default=True, help='메모리 최적화 사용')
    parser.add_argument('--use_mixed_precision', action='store_true', default=True, help='Mixed Precision 사용')
    parser.add_argument('--use_kv_cache_optimization', action='store_true', default=True, help='KV-Cache 최적화 사용')
    parser.add_argument('--temperature', type=float, default=0, help='생성 온도')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='최대 생성 토큰 수')
    parser.add_argument('--num_beams', type=int, default=1, help='빔 서치 개수')
    parser.add_argument('--benchmark', action='store_true', help='벤치마크 모드 (여러 번 실행)')
    parser.add_argument('--profile', action='store_true', help='프로파일링 모드 (성능 분석)')
    
    args = parser.parse_args()

    audio_messages = [{"role": "user", "content": "<speech>", "path": args.query_audio}]
    
    print("VocalNet 초기화 중...")
    print(f"최적화 설정:")
    print(f"  - 모델 컴파일: {args.use_compile}")
    print(f"  - Flash Attention: {args.use_flash_attention}")
    print(f"  - Mixed Precision: {args.use_mixed_precision}")
    print(f"  - KV-Cache 최적화: {args.use_kv_cache_optimization}")
    print(f"  - 메모리 최적화: {args.optimize_memory}")
    
    vocalnet = VocalNetModel(
        VOCALNET_MODEL,
        use_compile=args.use_compile,
        use_flash_attention=args.use_flash_attention,
        optimize_memory=args.optimize_memory,
        use_mixed_precision=args.use_mixed_precision,
        use_kv_cache_optimization=args.use_kv_cache_optimization,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams
    )
    
    print("모델 로딩 중...")
    vocalnet.__initilize__()
    
    if args.profile:
        print("\n=== 프로파일링 모드 ===")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            response = vocalnet.__call__(audio_messages)
        
        print("프로파일 결과:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        
    elif args.benchmark:
        print("\n=== 벤치마크 모드 ===")
        times = []
        for i in range(5):
            print(f"\n실행 {i+1}/5:")
            response = vocalnet.__call__(audio_messages)
            times.append(response['inference_time'])
            print(f"응답: {response['text'][:100]}...")
        
        print(f"\n=== 벤치마크 결과 ===")
        print(f"평균 추론 시간: {sum(times)/len(times):.3f}초")
        print(f"최소 추론 시간: {min(times):.3f}초")
        print(f"최대 추론 시간: {max(times):.3f}초")
        print(f"처리량 (TPS): {1.0/(sum(times)/len(times)):.2f} samples/sec")
    else:
        print("\n추론 실행 중...")
        response = vocalnet.__call__(audio_messages)
        print(f"\n응답: {response['text']}")
        print(f"추론 시간: {response['inference_time']:.3f}초")

    