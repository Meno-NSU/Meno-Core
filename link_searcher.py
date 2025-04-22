from lightrag import LightRAG
from pathlib import Path
import json


class LinkSearcher:

    def __init__(self, urls_path: Path | str, lightrag_instance: LightRAG, top_k: int, max_links: int = 3):
        self.lightrag_instance = lightrag_instance
        self.top_k = top_k
        urls_path = Path(urls_path)
        self.max_links = max_links
        with urls_path.open(mode='r', encoding='utf-8') as fp:
            self.urls = json.load(fp)

    async def get_links(self, query: str) -> list[str]:

        chunks_vdb = self.lightrag_instance.chunks_vdb
        text_chunks_db = self.lightrag_instance.text_chunks

        results = await chunks_vdb.query(query, top_k=self.top_k)
        if not len(results):
            return []
        chunks_ids = [r["id"] for r in results]
        chunks = await text_chunks_db.get_by_ids(chunks_ids)
        valid_chunks = [
            chunk for chunk in chunks if chunk is not None and "content" in chunk
        ]

        links = []
        for chunk in valid_chunks:
            content: str = chunk["content"]
            header = content[:content.find("\n")]
            link = self.urls.get(header)
            if link:
                links.append(self.urls[header])
        return links

    async def get_formated_answer(self, query: str, answer: str) -> str:
        links = await self.get_links(query)
        if links:
            return f"{answer}\n\nПолезные ссылки:\n{'\n'.join(links[:self.max_links])}"
        return answer
