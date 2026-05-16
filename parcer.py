import requests
import numpy as np
import time
import re
from collections import deque

BASE_URL = "https://api.openalex.org"
EMAIL = "malakhovs04@mail.ru"   

HEADERS = {"User-Agent": f"mailto:{EMAIL}"}

works_cache = {}                    
work_authors_cache = {}           


def transliterate_to_latin(text):
    if not text:
        return ""
    translit_dict = {
        'а':'a','б':'b','в':'v','г':'g','д':'d','е':'e','ё':'yo','ж':'zh','з':'z',
        'и':'i','й':'y','к':'k','л':'l','м':'m','н':'n','о':'o','п':'p','р':'r',
        'с':'s','т':'t','у':'u','ф':'f','х':'kh','ц':'ts','ч':'ch','ш':'sh',
        'щ':'shch','ъ':'','ы':'y','ь':'','э':'e','ю':'yu','я':'ya'
    }
    return ''.join(translit_dict.get(c.lower(), c) for c in text)


def normalize_name(name):
    if not name:
        return ""
    
    name = name.lower().strip()
    
    cyr = sum(1 for c in name if '\u0400' <= c <= '\u04FF' or c in 'ёЁ')
    alpha = sum(1 for c in name if c.isalpha())
    if cyr > 0.3 * alpha and alpha > 0:
        name = transliterate_to_latin(name)
    
    name = re.sub(r'[^a-z\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    if not name:
        return ""
    
    tokens = name.split()
    if not tokens:
        return ""
    
    lengths = [len(t) for t in tokens]
    max_len = max(lengths)
    for i in range(len(tokens)-1, -1, -1):
        if lengths[i] == max_len:
            surname = tokens[i]
            break
    
    given = [t for t in tokens if t != surname]
    if not given:
        return surname
    
    first_token = given[0]
    first_init = first_token[0]
    
    if len(first_token) > 2:
        second_init = ''
    else:
        second_init = given[1][0] if len(given) >= 2 else ''
    
    key = surname + first_init
    if second_init:
        key += second_init
    
    return key



def get_author_works(author_id):
    if author_id in works_cache:
        return works_cache[author_id]
    
    works = []
    page = 1
    while True:
        params = {
            "filter": f"authorships.author.id:https://openalex.org/{author_id},publication_year:2015-2026",
            "per-page": 200,
            "page": page,
            "select": "id,publication_year,authorships,referenced_works"
        }
        try:
            r = requests.get(f"{BASE_URL}/works", params=params, headers=HEADERS, timeout=15)
            if r.status_code != 200:
                break
        except requests.exceptions.RequestException:
            print("Ошибка соединения, пауза 5 сек...")
            time.sleep(5)
            continue
        
        data = r.json()
        results = data.get("results", [])
        if not results:
            break
        
        works.extend(results)
        page += 1
        time.sleep(0.35)
    
    works_cache[author_id] = works
    print(f"  {author_id} → {len(works)} работ за 2015–2026 гг.")
    return works


def get_work_authors(work_short_id):
    if work_short_id in work_authors_cache:
        return work_authors_cache[work_short_id]
    
    try:
        r = requests.get(f"{BASE_URL}/works/{work_short_id}", headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        authors = []
        for a in data.get("authorships", []):
            author_info = a.get("author")
            if author_info and author_info.get("id"):
                authors.append(author_info["id"].split("/")[-1])
        work_authors_cache[work_short_id] = authors
        time.sleep(0.25)
        return authors
    except:
        return []


def get_authors_bfs_group(start_author_id, limit, key_to_id, remap):
    authors = {}
    visited = set()
    queue = deque([start_author_id])

    while queue and len(authors) < limit:
        current_id = queue.popleft()
        if current_id in visited:
            continue
        visited.add(current_id)

        try:
            r = requests.get(f"{BASE_URL}/authors/{current_id}", headers=HEADERS, timeout=15)
            if r.status_code != 200:
                continue
        except:
            print("Ошибка соединения, пауза 5 сек...")
            time.sleep(5)
            continue

        author_json = r.json()
        display_name = author_json.get("display_name", "")
        normalized = normalize_name(display_name)

        if normalized in key_to_id:
            canonical = key_to_id[normalized]
            if current_id != canonical:
                remap[current_id] = canonical
                print(f"→ Объединён дубликат: {display_name} ({current_id}) → {canonical}")
            continue

        key_to_id[normalized] = current_id
        authors[current_id] = display_name
        print(f"{len(authors)} / {limit}  {display_name}  [{normalized}]")

        works = get_author_works(current_id)
        for work in works:
            for auth in work.get("authorships", []):
                a = auth.get("author")
                if a and a.get("id"):
                    co_id = a["id"].split("/")[-1]
                    if co_id not in visited and co_id not in queue:
                        queue.append(co_id)
        time.sleep(0.2)

    return authors


def build_coauthor_matrix(authors, remap):
    author_ids = list(authors.keys())
    n = len(author_ids)
    id_index = {author_ids[i]: i for i in range(n)}

    coauthor = np.zeros((n, n), dtype=int)

    print("\nСтроим матрицу соавторства")
    for i, author_id in enumerate(author_ids):
        print(f"{i+1}/{n}  {authors[author_id]}")
        works = get_author_works(author_id)
        for work in works:
            coauthors = []
            for a in work.get("authorships", []):
                author_info = a.get("author")
                if author_info and author_info.get("id"):
                    co_id = author_info["id"].split("/")[-1]
                    effective_id = remap.get(co_id, co_id)
                    coauthors.append(effective_id)

            for co_id in coauthors:
                if co_id in id_index and co_id != author_id:
                    j = id_index[co_id]
                    coauthor[i][j] += 1
                    coauthor[j][i] += 1

    coauthor_full = np.zeros((n + 1, n), dtype=int)
    coauthor_full[1:, :] = coauthor
    coauthor_full[0, :] = np.arange(1, n + 1)
    return coauthor_full



def build_citation_matrix(authors, remap):
    author_ids = list(authors.keys())
    n = len(author_ids)
    id_index = {author_ids[i]: i for i in range(n)}

    citation = np.zeros((n, n), dtype=int)

    print("\nСтроим матрицу цитирования")
    total_works = 0

    for i, author_id in enumerate(author_ids):
        print(f"{i+1}/{n}  {authors[author_id]}")
        works = get_author_works(author_id)
        total_works += len(works)
        
        for work in works:
            referenced = work.get("referenced_works", [])
            for ref_full in referenced:
                ref_short = ref_full.split("/")[-1]
                cited_authors = get_work_authors(ref_short)
                
                for ca in cited_authors:
                    effective = remap.get(ca, ca)
                    if effective in id_index:
                        j = id_index[effective]
                        citation[i][j] += 1

    print(f"Обработано работ: {total_works}")

    citation_full = np.zeros((n + 1, n), dtype=int)
    citation_full[1:, :] = citation
    citation_full[0, :] = np.arange(1, n + 1)
    return citation_full



def save_matrix(matrix, filename):
    np.savetxt(filename, matrix, fmt="%d")

def save_names(authors, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for i, name in enumerate(authors.values(), start=1):
            f.write(f"{i}: {name}\n")



if __name__ == "__main__":
    first_author = "1"
    second_author = "2"
    start_authors = [first_author, second_author]
    limits = [50, 50]

    authors_all = {}
    key_to_id = {}
    remap = {}

    for idx, start_id in enumerate(start_authors):
        print(f"\n=== Сбор от стартовой точки {idx+1} ===")
        group = get_authors_bfs_group(start_id, limits[idx], key_to_id, remap)
        authors_all.update(group)

    print(f"\nИтого уникальных авторов после слияния: {len(authors_all)}")

    co_matrix = build_coauthor_matrix(authors_all, remap)
    save_matrix(co_matrix, "coauthorship_matrix.txt")

    cit_matrix = build_citation_matrix(authors_all, remap)
    save_matrix(cit_matrix, "citation_matrix.txt")

    save_names(authors_all, "authors.txt")
