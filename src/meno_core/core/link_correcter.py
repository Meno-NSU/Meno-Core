# You may need to install the rapidfuzz library first:
# pip install rapidfuzz

import json
import re
from pathlib import Path
from typing import Optional, Tuple

# Using rapidfuzz for efficient Levenshtein distance calculation
from rapidfuzz.distance import Levenshtein
from rapidfuzz.process import extractOne


class LinkCorrecter:

    def __init__(self, urls_path: Path | str, dist_threshold: float):
        urls_path = Path(urls_path)
        self.dist_threshold = dist_threshold
        with urls_path.open(mode='r', encoding='utf-8') as fp:
            self.urls = set(json.load(fp).values())

    def find_closest_link(
        self,
        url: str,
        scorer=Levenshtein.normalized_similarity
    ) -> Optional[Tuple[str, float]]:
        """
        Finds the best matching URL from a set of valid links.

        Args:
            url: The URL to find a match for.
            link_set: A set of valid URLs to search within.
            scorer: The scoring function from rapidfuzz to use.
                    Defaults to normalized Levenshtein similarity.

        Returns:
            A tuple containing the best matching URL and its similarity score (0-1),
            or None if the link_set is empty.
        """
        
        # extractOne returns the best match in the format (choice, score, index)
        # The score is a similarity score from 0-100, so we divide by 100.
        result = extractOne(url, self.urls, scorer=scorer)
        if result:
            best_match, score, _ = result
            return best_match, score / 100.0
        return None

    async def replace_markdown_links(
        self,
        text: str,
    ) -> str:
        """
        Finds all Markdown-style links in a text, validates them against a set of
        known links, and replaces them if necessary.

        - If a link is already in the valid_links set, it is left unchanged.
        - If a link is not in the set, it's compared against the set. If a
        sufficiently similar link is found (based on the threshold), the
        original link is replaced with the closest match.
        - If no similar link is found, the URL part is removed, leaving only the
        link label (e.g., '[label]').

        Args:
            text: The input string containing markdown text.
            valid_links: A set of known, valid URLs.
            similarity_threshold: The minimum normalized Levenshtein similarity
                                (0.0 to 1.0) required to consider a URL a match.

        Returns:
            The processed text with links corrected or stripped.
        """
        # Regex to find markdown links: [label](url)
        # It captures the label (group 1) and the url (group 2)
        markdown_link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')

        def replacer(match: re.Match) -> str:
            """This function is called for each link found by re.sub."""
            label = match.group(1)
            url = match.group(2)

            # Case 1: The link is already valid.
            if url in self.urls:
                return match.group(0)  # Return the original [label](url) string

            # Case 2: The link is not valid, find the closest match.
            closest_match = self.find_closest_link(url)
            print(closest_match)

            # Case 3: A sufficiently close match was found.
            if closest_match and closest_match[1] >= self.dist_threshold:
                best_url, score = closest_match
                # print(f"Replacing '{url}' with '{best_url}' (Similarity: {score:.2f})")
                return f'[{label}]({best_url})'
            
            # Case 4: No close match was found.
            else:
                # print(f"No close match for '{url}'. Removing URL.")
                return label

        return markdown_link_pattern.sub(replacer, text)

# --- Example Usage ---
if __name__ == "__main__":
    # The set of "correct" or "allowed" URLs
    known_links = {
        "https://example.com/pages/about-us",
        "https://example.com/products/all",
        "https://example.com/contact",
        "https://example.com/blog/latest-posts"
    }

    # The text containing various links to be processed
    document_text = """
    Конечно! Чтобы ознакомиться со всеми специализациями и направлениями подготовки Новосибирского государственного университета, вам стоит посетить официальный сайт НГУ. Там есть раздел, посвященный учебной деятельности, где представлены все факультеты и их программы:

1. Основной ресурс:
   - [Сайт НГУ](https://www.nsu.ru/education/programs/)
   
2. Специализированные страницы:
   - Экономический факультет: [Ссылка](https://www.nsu.ru/economics/programms/)
   - Медико-психологический факультет: [Ссылка](https://www.nsu.ru/medicine/programms/)
   - Факультет журналистики: [Ссылка](https://www.nsu.ru/journalism/programms/)

Также на сайте НГУ можно найти информацию о новых направлениях, таких как:
- Гендерные исследования
- Информационные технологии в социологии
- Клиническая и организационная психология

Для более детального изучения каждого направления, рекомендуется также ознакомиться с описанием образовательных программ, учебными планами и аннотациями к дисциплинам, которые доступны непосредственно на страницах соответствующих факультетов.

Не забудьте проверить актуальность информации, так как НГУ регулярно обновляет свои образовательные программы и может внедрять новые специализации.
    """
    corr = LinkCorrecter("../../../resources/validated_urls.json", dist_threshold=0.85)
    print("--- Processing Document ---")
    processed_text = corr.replace_markdown_links(
        document_text,
    )
    print("\n--- Original Document ---")
    print(document_text)
    print("\n--- Processed Document ---")
    print(processed_text)
    
    # --- Example with a lower threshold ---
    corr = LinkCorrecter("../../../resources/validated_urls.json", dist_threshold=0.1)
    print("\n\n--- Processing with lower threshold (0.7) ---")
    processed_text_low_thresh = corr.replace_markdown_links(
        document_text,
    )
    print("\n--- Processed Document (Low Threshold) ---")
    print(processed_text_low_thresh)

