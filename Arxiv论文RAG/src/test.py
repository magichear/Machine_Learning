# %%
from Ask import Ask
from search import Search
from time import time

if __name__ == "__main__":
    """
    可逐个单元格调试     [UPDATE] 最新使用示例请移步 testAsk.ipynb, 本脚本已落后多个版本未经测试
    """
    search = Search()
    ask = Ask(search_engine=search)

    # %% =====================================将查询问题重写为英文学术表示========================================
    questions = []

    for i in range(50):
        identity, template = search.queryAutoGen(i)
        # 手动询问请使用queryGen方法替换queryAutoGen
        questions.append((identity, template))

    querys = []
    # %%
    for identity, template in questions:
        # print(f"Identity: {identity}")
        # print(f"{template}")
        # print("=" * 50)
        querys.append(ask.chat(identity, template))
        time.sleep(2)

    with open("extracted_querys.txt", "w", encoding="utf-8") as file:
        for query in querys:
            file.write(query.strip() + "\n\n")
    # %% =====================================检查回复格式是否严格符合要求，不符合可进行一些人工修整========================================
    with open("extracted_querys.txt", "r", encoding="utf-8") as file:
        querys = [line.strip() for line in file if line.strip()]
    print(len(querys))
    print(querys)
    # %% =====================================在知识库中检索相关内容，并提问========================================
    results = []
    i = 0
    for query in querys:
        results.append(ask.ask([query], [search.prompts.questions[i]]))
        time.sleep(5)
        i += 1

    with open("results.txt", "w", encoding="utf-8") as file:
        for result in results:
            file.write(result.strip() + "\n\n\n")
    # %% =====================================评分========================================
