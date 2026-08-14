import time
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


with open("/Users/yoshimurakenjin/かんじぃPT/違和感の類型化/漢字2_1行1漢字.txt")as f:
    kanji_list = f.read().splitlines()
grade = 2
age = 8
max_chars = 200
hero_name = "ゆうき"


def build_prompt(selected_kanji, grade, age, hero_name, max_chars, story_type=None):

    prompt = (
        f"あなたはプロの作家です。"
        f"ゴール「{selected_kanji}」という字を必ず使って、{grade}年生（{age}歳）でも楽しめる物語を作成します。"
        f"ルール1:「{hero_name}」という主人公がいる。"
        f"2:起承転結のある物語。"
        f"3:{max_chars}文字以内。"
    )

    return prompt


results = {} 

for kanji in kanji_list:
    try:
        prompt = build_prompt(kanji, grade, age, hero_name, max_chars)

        time.sleep(1)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        story = response.choices[0].message.content
        results[kanji] = story
        print(f"生成完了：{kanji}")

    except Exception as e:
        print(f"エラー発生：{kanji} → {e}")

import json
with open("stories.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
