# pip install arxiv
import arxiv
import os
import time


def fetch_arxiv_papers(query, max_results=10, save_dir="./datasets/arxiv/papers"):
    os.makedirs(save_dir, exist_ok=True)

    # 搜索论文
    search = arxiv.Search(
        query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate
    )

    # 下载 PDF
    for result in search.results():
        pdf_filename = f"{result.entry_id.split('/')[-1]}.pdf"
        try:
            result.download_pdf(dirpath=save_dir, filename=pdf_filename)
        except Exception as e:
            print(e)


start_time = time.time()
fetch_arxiv_papers(query="cat:cs.CL", max_results=20)

print(f"总耗时: {time.time() - start_time:.2f} 秒")
