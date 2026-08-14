import json
import os
import asyncio
import openai
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# =====================
# Initial settings
# =====================
load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_CONCURRENT_REQUESTS = 5

 改善1: モデルごとにセマフォを分ける
_semaphores: dict[str, asyncio.Semaphore] = {}

def get_semaphore(model_name: str) -> asyncio.Semaphore:
    if model_name not in _semaphores:
        _semaphores[model_name] = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    return _semaphores[model_name]

# =====================
# 1. SYSTEM_PROMPT
# =====================
SYSTEM_PROMPT = """
あなたは小学校の教員です。小学2年生を対象とした漢字学習支援に使用する文章の、教育的な妥当性を評価してもらいます。
以下の類型定義と判断基準に基づいて、物語の違和感を分類してください。

【類型】
C01: 語法の誤り
定義：一文単位で、文法や語の結びつき（コロケーション）が不自然。
判断基準：問題となる箇所を一般的なものに言い換えなければ物語が理解できない場合ここに分類。

C02: 因果ねじれ
定義：ある結果とその原因として記されているものに関係がない。
判断基準：行動や出来事の原因として書かれているものに納得がいかない場合ここに分類。

C03: 急展開
定義：描写をつなぐプロセスが抜け落ちていて、読み手が動作や出来事を想像する必要がある。
判断基準：行動や出来事は世界観に一致しているが、なぜそうなるのかわからない場合ここに分類。

C04: 設定の不整合
定義：物語の途中で設定やキャラクターの行動に矛盾が生じている。
判断基準：物語の設定（ルール）に反する行動や展開があり辻褄が合わない場合ここに分類。

C05: 接続・品詞
定義：文脈に適した接続や品詞を使用できていない
判断基準：物語を通して読んだ時に、文と文を接続する単語や、品詞が自然でない場合ここに分類。

C06: 主述の明示
定義：主語と述語の繋がりや記述が不明瞭で、誰が何をしているのかが分かりにくい。
判断基準：物語を通して読んでも、誰の動作か不明で意味が通じない場合ここに分類。

C07: 教育的観点
定義：文章としては問題ないが漢字学習の支援にはならない
判断基準：C01~C06に該当せず文章としては問題ないが、指定漢字の意味を生かした文章でない場合ここに分類。

NONE: 違和感がなく、物語として自然。

【カウントルール】
- 複数の異なる類型の違和感がある場合、それぞれ1回ずつ出力する
- 同一類型は1回のみ
- ラベルは重複なし・昇順で出力

【重要ルール】
- NONEは単独の場合のみ出力すること
- 他のラベルとNONEを同時に出力してはならない
- 出力は指定されたJSON形式のみとする
- まず判定理由を述べてからラベルを出力する
"""

# =====================
# 2. User Prompt Generation
# =====================
def make_user_prompt(story_text: str) -> str:
    return f"""
以下の物語を分類してください。

【使用可能ラベル】
["C01","C02","C03","C04","C05","C06","C07","NONE"]

【例】
文章：そして、冒険が終わると、ゆうきは自分の心に「活」の大切さを刻みました。
判定理由：「活の大切さを心に刻む」という言い方は不自然で、語のつながりに誤りがある。
出力：["C01"]

文章：心配したゆうきは石に語りかけ、「心を大切にするね」と約束しました。すると石は再び輝き、嵐は止まりました。
判定理由：「心を大切にする」ことと「嵐が止まる」ことに因果関係はない。
出力：["C02"]

文章：ある朝、道がぐねぐねにまがっていた。なきそうなとき、木が「うたって」とささやく。ゆうきがうたうと道はまっすぐ。
判定理由：ゆうきが歌うと道がまっすぐになる理由が省略されていて、理解できない。
出力：["C03"]

文章：その石を握ると、動物たちと話せる魔法の力を手に入れました。毎日、ゆうきは動物たちと楽しく遊び、町の人々も笑顔になりました。しかし、ある夜、ゆうきは石が消えているのに気付き、悲しくなります。でも次の日、動物たちは「心が通じ合っている限り、いつでも話せるよ」と教えてくれました。
判定理由：石が消えたなら、動物はゆうきに言葉で教えることはできないはず。
出力：["C04"]

文章：「これに絵を描こう！」と思い、早速絵を描き始めました。しかし、描いていると、どんどん色が光り出し、キャンバスから不思議な世界が現れました。
判定理由：「しかし」を使っているが、内容が対立したり転換したりしていない。
出力：["C05"]

文章：ゆうきは森で『理』と光る小石を見つけた。『ひとつだけ願いをかなえる』とある。
判定理由：物語の内容に対して『理』という漢字を使う必要性がない。
出力：["C07"]

文章：今日は楽しかった。
判定理由：問題なし。
出力：["NONE"]

【出力形式】
{{
  "results": [
    {{
      "reason": "判定理由",
      "labels": ["<ラベル>"]
    }}
  ]
}}

【指示】
- JSONのみ出力すること。ルートは必ず "results" キーを持つオブジェクトにすること。
- 必ず「reason」に判定理由を簡潔に書くこと
- その後にlabelsを出力すること
- 各データに対して1つのlabels配列を返すこと。

【データ】
{story_text}
"""

# =====================
# 3. Post-processing logic
# =====================
def normalize_labels(labels: list) -> list[str]:
    """ラベルの表記揺れを補正し、NONEルールを適用し、ソートする"""
    if not isinstance(labels, list):
        return ["ERROR"]

    valid_labels = {"C01", "C02", "C03", "C04", "C05", "C06", "C07", "NONE"}
    cleaned = {str(l).strip().upper() for l in labels}
    final = [l for l in cleaned if l in valid_labels]

    # 改善4: NONEと他ラベルの混在を修正
    if len(final) > 1 and "NONE" in final:
        final.remove("NONE")

    return sorted(final) if final else ["NONE"]

# =====================
# 4. Asynchronous classification engine
# =====================
async def classify_story_async(
    story_id: str,
    story_text: str,
    model_name: str,
) -> dict:
    # 改善5: 空テキストのガード
    if not story_text or not str(story_text).strip():
        return {
            "id": story_id,
            "labels": ["ERROR"],
            "reason": "空のテキストです",
            "status": "skip",
        }

    # 改善1: モデルごとのセマフォを使用
    semaphore = get_semaphore(model_name)

    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": make_user_prompt(story_text)},
                    ],
                    response_format={"type": "json_object"},
                    timeout=60.0,
                )

                content = response.choices[0].message.content.strip()

                # JSONエラーの可視化
                try:
                    data = json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"[JSON ERROR] id={story_id} model={model_name}\n{content}")
                    raise

                results_list = data.get("results", [])
                if not results_list:
                    raise ValueError("LLMレスポンスにresultsがありません")

                res_obj = results_list[0]
                labels = normalize_labels(res_obj.get("labels", []))
                reason = res_obj.get("reason", "理由の取得に失敗しました")

                return {
                    "id": story_id,
                    "labels": labels,
                    "reason": reason,
                    "model": model_name,       # ログ強化
                    "raw_response": content,   # ログ強化
                    "status": "success",
                }

            # 改善6: エラー種別ごとのハンドリング
            except openai.AuthenticationError as e:
                # 認証エラーはリトライ不要
                return {
                    "id": story_id,
                    "labels": ["ERROR"],
                    "reason": f"認証エラー: {e}",
                    "status": "error",
                }
            except openai.RateLimitError:
                if attempt == 2:
                    return {
                        "id": story_id,
                        "labels": ["ERROR"],
                        "reason": "レート制限: リトライ上限到達",
                        "status": "error",
                    }
                wait_time = (2 ** attempt) + 1
                await asyncio.sleep(wait_time)
            except Exception as e:
                if attempt == 2:
                    return {
                        "id": story_id,
                        "labels": ["ERROR"],
                        "reason": f"最終エラー: {e}",
                        "status": "error",
                    }
                wait_time = (2 ** attempt) + 1
                await asyncio.sleep(wait_time)

# =====================
# 5. Execute experiment
# =====================
async def process_experiment(
    input_path: str,
    model_name: str,
    output_path: str,
) -> None:
    if not os.path.exists(input_path):
        print(f"❌ Skip (入力ファイルなし): {input_path}")
        return

    # 改善3: 出力ファイルが既存の場合はスキップ
    if os.path.exists(output_path):
        print(f"⏭️  Skip (出力済み): {output_path}")
        return

    with open(input_path, encoding="utf-8") as f:
        stories_dict = json.load(f)

    print(f"🔥 Start: {model_name} / {input_path}")

    tasks = [
        classify_story_async(k, v, model_name)
        for k, v in stories_dict.items()
    ]

    # 改善7: tqdmにラベルを追加
    desc = f"{model_name} / {os.path.basename(input_path)}"
    results = await tqdm.gather(*tasks, desc=desc)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"✅ Done: {output_path}")

# =====================
# 6. Main
# =====================
async def main() -> None:
    EXPERIMENTS = [
        ("/Users/yoshimurakenjin/かんじぃPT/類型化の自動化/jsonファイル/各モデルの物語/4omini-stories.json", "gpt-4o",  "result_4o_on_4omini.json"),
        ("/Users/yoshimurakenjin/かんじぃPT/類型化の自動化/jsonファイル/各モデルの物語/4omini-stories.json", "gpt-5",   "result_5_on_4omini.json"),
        ("/Users/yoshimurakenjin/かんじぃPT/類型化の自動化/jsonファイル/各モデルの物語/4o-stories.json", "gpt-4o", "result_4o_on_4o.json"),
        ("/Users/yoshimurakenjin/かんじぃPT/類型化の自動化/jsonファイル/各モデルの物語/4o-stories.json", "gpt-5", "result_5_on_4o.json"),
        ("/Users/yoshimurakenjin/かんじぃPT/類型化の自動化/jsonファイル/各モデルの物語/5-stories.json", "gpt-4o", "result_4o_on_5.json"),
        ("/Users/yoshimurakenjin/かんじぃPT/類型化の自動化/jsonファイル/各モデルの物語/5-stories.json", "gpt-5", "result_5_on_5.json"),
        
        # 他の組み合わせをここに追加
    ]

    # 改善2: 実験を並列実行
    await asyncio.gather(*[
        process_experiment(inp, model, outp)
        for inp, model, outp in EXPERIMENTS
    ])

if __name__ == "__main__":
    asyncio.run(main())
