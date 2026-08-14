import time
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# =====================
# 1. Initial settings
# =====================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("/Users/yoshimurakenjin/かんじぃPT/違和感の類型化/漢字2_1行1漢字.txt") as f:
    kanji_list = f.read().splitlines()

GRADE = 2
AGE = 8
MAX_CHARS = 200
HERO_NAME = "ゆうき"
OUTPUT_PATH = "4omini-few-shot.json"


# =====================
# 2. Prompt Generation
# =====================
def build_prompt(selected_kanji, grade, age, hero_name, max_chars, story_type=None):
    """指定された漢字を使った物語生成用プロンプトを組み立てる"""
    prompt = (
        f"あなたはプロの作家です。"
        f"ゴール「{selected_kanji}」という字を必ず使って、{grade}年生（{age}歳）でも楽しめる物語を作成します。"
        f"ルール1:「{hero_name}」という主人公がいる。"
        f"2:起承転結のある物語。"
        f"3:{max_chars}文字以内。"
    )
    return prompt


# =====================
# 3.Story generation engine (synchronous, one at a time)
# =====================
def generate_story(kanji: str) -> str:
    """1つの漢字に対して物語を1件生成する"""
    prompt = build_prompt(kanji, GRADE, AGE, HERO_NAME, MAX_CHARS)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


# =====================
# 4. Execution loop
# =====================
def main() -> None:
    results = {}

    for kanji in kanji_list:
        try:
            story = generate_story(kanji)
            results[kanji] = story
            print(f"生成完了：{kanji}")

            # APIのレートリミットを回避するための待機
            time.sleep(1)

        except Exception as e:
            print(f"エラー発生：{kanji} → {e}")

# =====================
# 5. Save Results (JSON)
# =====================
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ 保存完了：{OUTPUT_PATH}")


# =====================
# 6. Main
# =====================
if __name__ == "__main__":
    main()
