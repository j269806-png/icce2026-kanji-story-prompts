import time
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

# Load the API key from the .env file
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Conditions for Story Generation
GRADE = 2
AGE = 8
MAX_CHARS = 200
HERO_NAME = "ゆうき"
KANJI_FILE = "漢字2_1行1漢字.txt"
OUTPUT_FILE = "4omini-zero-shot.json"
MODEL = "gpt-4o-mini"


def load_kanji_list(path: str) -> list[str]:
    """漢字リストをファイルから読み込む"""
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


def build_prompt(selected_kanji: str, grade: int, age: int, hero_name: str, max_chars: int) -> str:
    """物語生成用のプロンプトを作成する"""
    return (
        f"あなたはプロの作家です。"
        f"ゴール「{selected_kanji}」という字を必ず使って、"
        f"{grade}年生（{age}歳）でも楽しめる物語を作成します。"
        f"ルール1:「{hero_name}」という主人公がいる。"
        f"2:起承転結のある物語。"
        f"3:{max_chars}文字以内。"
    )


def generate_story(kanji: str) -> str:
    """指定した漢字を含む物語を1件生成する"""
    prompt = build_prompt(kanji, GRADE, AGE, HERO_NAME, MAX_CHARS)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def main() -> None:
    kanji_list = load_kanji_list(KANJI_FILE)
    results: dict[str, str] = {}

    for kanji in kanji_list:
        try:
            time.sleep(1)  # APIのレートリミットを考慮
            results[kanji] = generate_story(kanji)
            print(f"生成完了：{kanji}")
        except Exception as e:
            print(f"エラー発生：{kanji} → {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
