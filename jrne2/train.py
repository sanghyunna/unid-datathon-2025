import os
import json
from glob import glob
from typing import List, Dict, Any
from collections import Counter
import matplotlib.pyplot as plt
import koreanize_matplotlib     # 👈 1. 한글 폰트 import
from wordcloud import WordCloud
from multiprocessing import Pool # 👈 2. 최적화(병렬 처리)
import os                      # 👈 2. 최적화(CPU 코어 수)
from tqdm import tqdm          # 👈 3. 진행률 표시
from transformers import pipeline  # Hugging Face Transformers 라이브러리 추가

# --- (1/5) Okt 로더 및 토크나이저 정의 ---

# Okt 객체는 로딩 시간이 걸리므로 스크립트 실행 시 한 번만 생성합니다.
try:
    from konlpy.tag import Okt
    okt = Okt()
except Exception as e:
    print(f"KoNLPy(Okt) 로딩 실패. 1. 'pip install konlpy' 2. Java 설치 및 JAVA_HOME 환경변수 확인 필요. 오류: {e}")
    exit()

# [Before] 베이스라인의 simple_tokenize 함수
def simple_tokenize(s: str) -> List[str]:
    s = (s or "")
    s = s.replace("##", " ").replace(",", " ").replace("(", " ").replace(")", " ")
    s = s.replace(":", " ").replace("?", " ").replace("!", " ").replace("·", " ")
    return [t for t in s.strip().split() if t]

# [After] 우리가 개선한 new_tokenize 함수
# ❗ 1. 제거할 품사 태그 정의 (불용어 태그)
STOP_TAGS = ['Josa', 'Punctuation', 'Suffix', 'Eomi', 'Verb']

# ❗ 2. 제거할 단어 정의 (불용어)
STOP_WORDS = ['있다', '하다', '같다', '어디', '대해', '알리다', '보이다', '알다'] 

def new_tokenize(s: str) -> list[str]:
    """Okt를 사용해 문장에서 핵심 키워드(명사 등)만 추출합니다."""
    s = (s or "")
    if not s:
        return []
    try:
        pos_result = okt.pos(s, norm=True, stem=True)
    except Exception:
        return []

    keywords = []
    for word, tag in pos_result:
        if tag in STOP_TAGS or word in STOP_WORDS:
            continue
        if len(word) > 1: # 한 글자 단어 제거 (예: '경', '제')
            keywords.append(word)
    return keywords

# --- (2/5) JSON 데이터 로딩 함수 ---

def find_jsons(json_dir: str) -> List[str]:
    """지정된 디렉토리에서 모든 .json 파일 경로를 찾습니다. (하위 폴더 포함)"""
    if os.path.isdir(json_dir):
        # recursive=True로 하위 폴더의 모든 json을 검색
        return sorted(glob(os.path.join(json_dir, "**", "*.json"), recursive=True))
    raise FileNotFoundError(f"json_dir not found: {json_dir}")

def read_json(path: str) -> Dict[str, Any]:
    """JSON 파일을 읽어 딕셔너리로 반환합니다."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def summarize_text(text: str, max_length: int = 50, min_length: int = 10) -> str:
    """
    Hugging Face Transformers의 요약 파이프라인을 사용하여 텍스트를 요약합니다.
    """
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"[요약 오류] {e}")
        return text  # 요약 실패 시 원본 텍스트 반환

def extract_all_instructions_and_answers(json_dir_list: List[str]) -> List[Dict[str, str]]:
    """
    주어진 *폴더 리스트*의 모든 JSON 파일을 순회하며
    'visual_instruction'과 'visual_answer'를 추출하여 리스트로 반환합니다.
    """
    all_data = []
    print("=== Instruction 및 Answer 추출 시작 ===")
    for folder_path in json_dir_list:
        print(f"--- '{folder_path}' 폴더 검색 중... ---")
        try:
            json_files = find_jsons(folder_path)
            print(f"총 {len(json_files)}개의 JSON 파일 발견.")
        except FileNotFoundError as e:
            print(f"[오류] {e}")
            continue

        # tqdm으로 개별 파일 처리 진행률 표시
        for json_path in tqdm(json_files, desc=f"  -> {os.path.basename(folder_path)} 처리 중", leave=False):
            try:
                data = read_json(json_path)
                annotations = data.get("learning_data_info", {}).get("annotation", [])
                if not annotations:
                    continue 

                for ann in annotations:
                    instruction = ann.get("visual_instruction")
                    answer = ann.get("visual_answer")
                    if instruction:
                        summarized_answer = summarize_text(answer) if answer else ""
                        all_data.append({
                            "instruction": instruction,
                            "answer": summarized_answer
                        })
            except Exception:
                pass  # 개별 파일 오류는 조용히 건너뛰기
    return all_data

# --- (3/5) 시각화 함수 (워드 클라우드) ---

def plot_word_cloud(tokens: List[str], save_path: str = "wordcloud_after.png"):
    """
    (After 기준) 추출된 토큰 리스트로 워드 클라우드를 생성하고 파일로 저장합니다.
    """
    print(f"\n--- 워드 클라우드 생성 중... ({save_path}) ---")
    if not tokens:
        print("[경고] 워드 클라우드를 만들 토큰이 없습니다.")
        return

    counts = Counter(tokens)
    # 폰트 경로를 수동 지정(malgun.ttf)하거나, koreanize_matplotlib 것을 사용
    font_path = koreanize_matplotlib.get_font_path()
    # font_path = "C:/Windows/Fonts/malgun.ttf" # 윈도우 사용자 수동 지정 예시
    
    wc = WordCloud(
        font_path=font_path,
        width=800,
        height=600,
        background_color="white",
        max_words=100
    )
    wc.generate_from_frequencies(counts)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(save_path)
    print(f"✅ 워드 클라우드가 '{save_path}'에 저장되었습니다.")
    plt.close()

# --- (4/5) 시각화 함수 (Before-After 비교 차트) ---

def plot_before_after_chart(
    before_tokens: List[str], 
    after_tokens: List[str], 
    n: int = 20, 
    save_path: str = "keywords_compare.png"
):
    """
    [Before]와 [After]의 상위 N개 키워드를 나란히 바 차트로 그려 저장합니다.
    """
    print(f"\n--- Before-After 상위 {n}개 키워드 비교 차트 생성 중... ---")
    if not before_tokens or not after_tokens:
        print("[경고] 비교 차트를 만들 토큰이 없습니다.")
        return

    # 1. (Before) 상위 N개 키워드 추출
    before_common = Counter(before_tokens).most_common(n)
    before_common.reverse()
    before_labels = [item[0] for item in before_common]
    before_freqs = [item[1] for item in before_common]

    # 2. (After) 상위 N개 키워드 추출
    after_common = Counter(after_tokens).most_common(n)
    after_common.reverse()
    after_labels = [item[0] for item in after_common]
    after_freqs = [item[1] for item in after_common]

    # 3. 1x2 (가로 2칸) 서브플롯 생성
    fig, axes = plt.subplots(1, 2, figsize=(20, 10)) # 1줄 2칸
    
    # 4. [Before] 차트 그리기 (왼쪽: axes[0])
    axes[0].barh(before_labels, before_freqs, color='royalblue')
    axes[0].set_title('Before: Baseline (simple_tokenize)', fontsize=16)
    axes[0].set_xlabel('빈도수')
    axes[0].set_ylabel('토큰')
    axes[0].grid(axis='x', linestyle='--', alpha=0.7)

    # 5. [After] 차트 그리기 (오른쪽: axes[1])
    axes[1].barh(after_labels, after_freqs, color='darkviolet')
    axes[1].set_title('After: Improved (KoNLPy + Stopwords)', fontsize=16)
    axes[1].set_xlabel('빈도수')
    axes[1].set_ylabel('핵심 키워드')
    axes[1].grid(axis='x', linestyle='--', alpha=0.7)

    # 6. 전체 제목 및 레이아웃 설정
    fig.suptitle(f'Instruction 키워드 분석 (Top {n}): Before vs After', fontsize=20, y=1.03)
    plt.tight_layout()
    
    # 7. 파일로 저장
    plt.savefig(save_path)
    print(f"✅ 비교 차트가 '{save_path}'에 저장되었습니다.")
    plt.close()

# --- (5/5) 스크립트 메인 실행 ---
if __name__ == "__main__":
    
    # ❗❗ 여기에 분석할 모든 JSON 폴더 경로를 리스트로 넣어주세요.
    JSON_DIR_LIST = [
        "C:/Users/jrne/Desktop/train_valid/train/press_json",
        "C:/Users/jrne/Desktop/train_valid/train/report_json"
        # "또 다른 폴더 경로가 있다면 여기에 추가..."
    ]
    
    # 1. 모든 instruction 및 answer 추출
    all_data = extract_all_instructions_and_answers(JSON_DIR_LIST)
    
    if not all_data:
        print("="*40)
        print("❌ 분석할 instruction과 answer를 찾지 못했습니다. JSON_DIR_LIST 경로를 확인해주세요.")
        print("="*40)
    else:
        print("\n" + "="*40)
        print(f"✅ 총 {len(all_data)}개의 instruction 및 answer 수집 완료. 토큰화 시작...")
        print("="*40)

        # 2. [Before]와 [After] 토큰 리스트를 *병렬*로 생성
        
        # CPU 코어 수 확인 (최대 8개 or 현재 코어 수)
        num_cores = min(os.cpu_count() or 4, 8) 
        print(f"--- {num_cores}개의 CPU 코어를 사용하여 병렬 처리 시작 ---")

        # [Before] 토큰화 (simple_tokenize는 매우 빠르므로 굳이 병렬처리 안 함)
        before_tokens = []
        for data in all_data:
            before_tokens.extend(simple_tokenize(data["instruction"] + " " + data["answer"]))
        print("--- [Before] 토큰화 완료 (fast) ---")
        
        # [After] 토큰화 (new_tokenize는 매우 느리므로 병렬 처리)
        after_tokens_list = []  # [[t1, t2], [t3, t4], ...] 형태의 리스트
        
        # 3. Pool 객체를 생성하여 new_tokenize 함수를 병렬 실행
        with Pool(processes=num_cores) as pool:
            # pool.imap: 함수를 데이터에 매핑. tqdm으로 진행률 표시
            after_tokens_list = list(tqdm(
                pool.imap(new_tokenize, [data["instruction"] + " " + data["answer"] for data in all_data], chunksize=100),  # 👈 chunksize 추가
                total=len(all_data),
                desc="[After] 토큰화 중"
            ))
        
        # 4. 병렬 처리된 결과를 하나의 리스트로 펼치기 (Flatten)
        after_tokens = [token for sublist in after_tokens_list for token in sublist]
        
        print("--- 병렬 처리 완료. 시각화 시작... ---")

        # 5. 시각화 함수 호출
        
        # (시각화 1) Before-After 비교 바 차트
        plot_before_after_chart(
            before_tokens, 
            after_tokens, 
            n=20, 
            save_path="keywords_compare.png"
        )
        
        # (시각화 2) After 기준 워드 클라우드
        plot_word_cloud(after_tokens, save_path="wordcloud_after.png")

        print("\n" + "="*40)
        print("✨ 모든 시각화 자료 생성이 완료되었습니다. (png 파일 확인)")
        print("="*40)