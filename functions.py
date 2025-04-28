import requests
import csv
import os
import glob
from pyspark.sql.functions import monotonically_increasing_id, row_number
import requests
from bs4 import BeautifulSoup as bs
import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import count
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from pyspark.ml.clustering import PowerIterationClustering

# Some domains tell nothing about the actual source,
# like archive links or a paper's DOI, so they should be filtered out
BLACKLIST = ["doi.org", "web.archive.org", "creativecommons.org", "search.worldcat.org"]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
}

FILEPATH_TITLES_CSV     = 'titles.csv'
FILEPATH_REFERENCES_CSV = 'references.csv'
FILEPATH_EDGES_RAW_CSV  = 'edges_raw.csv'
FILEPATH_EDGES_CSV      = 'edges.csv'
FILEPATH_VERTICES_CSV   = 'vertices.csv'

spark = SparkSession.builder \
    .appName("YourAppName") \
    .config("spark.eventLog.enabled", "false") \
    .getOrCreate()

def hash_func(x):
    return hash(x)

UDF_HASH = udf(hash_func, IntegerType())

def read_data_to_dataframe(file_path):
    """Reads a CSV file into a Spark DataFrame with headers."""
    df = spark.read.csv(file_path, header=True)

    return df

def write_list_to_csv(my_list, file_path, col1, col2):
    """Appends a list of tuples to a CSV file, adding headers if file doesn't exist.
    Each tuple becomes a row with columns col1 and col2."""
    # Check if file exists to determine if we need headers
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)

        # Write headers if file doesn't exist
        if not file_exists:
            writer.writerow([col1, col2])

        writer.writerows(my_list)

    print(f"List appended to {file_path}")

def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def write_titles_csv(keywords):
    """Searches Wikipedia for articles matching keywords and saves their IDs/titles to CSV.
    Makes multiple API requests to get comprehensive results."""

    base_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": keywords,
        "srlimit": 10,                      # Dont forget to change it
        "format": "json"
    }

    i = 2                                   # Dont forget to change it
    while i > 0:
        titles = []

        response = requests.get(base_url, params=params).json()
        search_results = response.get("query", {}).get("search", [])
        # Store both the page id and and title as a tuple
        titles.extend([(result["pageid"], result["title"]) for result in search_results])

        write_list_to_csv(titles, FILEPATH_TITLES_CSV, "title_id", "title")
        # Example output:
        # title_id,title
        # 5042951,Climate change
        # 30242372,Paris Agreement
        # 12474403,Climate change denial

        if "continue" in response:
            params["sroffset"] = response["continue"]["sroffset"]
        else:
            break

        i -= 1

def get_wikipedia_html(title):
    """Fetches the HTML content of a Wikipedia article by title.
    Returns raw HTML string or None if request fails."""

    url = "https://en.wikipedia.org/api/rest_v1/page/html/{title}".format(title=title)

    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()                     # Check for HTTP errors
        return response.text                            # Returns raw HTML
    except requests.exceptions.RequestException as e:
        print(f"Error fetching HTML for '{title}': {e}")
        return None

def extract_references(html_content):
    """Parse the html content to get the article references"""
    soup = bs(html_content, 'html.parser')
    ref = soup.find('ol', class_="references")
    if ref:
        return list(ref.children)
    return None

def extract_refs_per_title(refs_for_one_article, title_id):
    """Extracts and filters website references from article references.
    Saves valid (title_id, domain) pairs to CSV, excluding blacklisted domains."""

    list_final_hrefs_per_title = []
    for ref in refs_for_one_article:
        if len(ref) > 1:
            a_refs = ref.find_all('a')
            for a in a_refs:
                if a.has_attr('href'):
                    href = a['href']
                    # Get only the tags that contain website links
                    match_ = re.match(r'^https?://', href)
                    if match_:
                        # Extract only the domain
                        href_without_prefix = re.sub(r'^https?://', '', href)
                        href_without_prefix = href_without_prefix.split('/')[0]
                        # Make sure all the letters in the domain lowercase
                        final_href = href_without_prefix.lower()

                        # Ensure the link is not in the bloacklist
                        if final_href not in BLACKLIST:
                            list_final_hrefs_per_title.append((title_id, final_href))

    # Saving the refs for one title at a time
    write_list_to_csv(list_final_hrefs_per_title, FILEPATH_REFERENCES_CSV, "title_id", "ref")
    # Example output:
    # title_id,ref
    # 5042951,data.giss.nasa.gov
    # 5042951,ui.adsabs.harvard.edu
    # 5042951,api.semanticscholar.org


def write_refs_csv():
    """Processes titles CSV to extract and save references for each Wikipedia article.
    Reads input file and writes results incrementally so as not to bring the
    entire dataset in memory."""
    with open(FILEPATH_TITLES_CSV, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            title_id, title = row[0], row[1]
            # Extract all the references from one article html page at a time
            refs_for_one_article = extract_references(get_wikipedia_html(title))
            if refs_for_one_article:
                extract_refs_per_title(refs_for_one_article, title_id)


def write_edges_raw_csv():
    """Generates undirected edges between references from the same article.
    Creates all possible reference pairs per article, sorted alphabetically."""

    df = read_data_to_dataframe(FILEPATH_REFERENCES_CSV)
    df = df.withColumn("title_id", df["title_id"].cast("integer"))
    df = df.dropDuplicates().orderBy('title_id').dropna()
    unique_title_ids = [row.title_id for row in df.select("title_id").distinct().collect()]

    for title_id in unique_title_ids:
        edges_per_title = []
        refs_per_title = [row.ref for row in df.filter(df.title_id == title_id).collect()]
        # Iterate through all the the possible reference combinations
        for i in range(len(refs_per_title)):
            for j in range(i + 1, len(refs_per_title)):
                # To ensure consistency, the smaller references — sorted in
                # alphabetical order — are stored first
                if refs_per_title[i] < refs_per_title[j]:
                    edge = (refs_per_title[i], refs_per_title[j])
                else:
                    edge = (refs_per_title[j], refs_per_title[i])
                edges_per_title.append(edge)

        # Writing data to a csv file
        write_list_to_csv(edges_per_title, FILEPATH_EDGES_RAW_CSV, "src", "dst")
        # Example output:
        # src,dst
        # www.nytimes.com,www.theguardian.com
        # eprints.lse.ac.uk,www.nytimes.com
        # www.ipcc.ch,www.nytimes.com


def write_edges_and_vertices_csv(num_clusters, max_iterations):
    """Processes raw edges into weighted edges and clusters references using PIC.
    Outputs two CSVs: weighted edges and vertices with cluster assignments."""
    edges_df = read_data_to_dataframe(FILEPATH_EDGES_RAW_CSV)

    # Create a new dataframe that contains the weights of the edges.
    # The links are hashed because the PowerIterationClustering module only accepts numeric node values
    new_edges_df = edges_df.groupBy(edges_df.columns).agg(count("*").alias("weight"))
    new_edges_df = new_edges_df.withColumn("src", UDF_HASH(new_edges_df["src"])) \
        .withColumn("dst", UDF_HASH(new_edges_df["dst"]))

    # Write the data to a new csv file. This one contains the weights of the edges
    new_edges_df.coalesce(1).write.csv(".", header=True, mode="overwrite")
    os.rename(glob.glob("part*.csv")[0], FILEPATH_EDGES_CSV)
    # Example output:
    # src,dst,weight
    # 548994724,-1558522398,1
    # -512756871,548994724,2
    # 1532437153,-1268887550,2

    # Create a new vertices dataframe. To each node there will be assigned a cluster
    vertices_df = edges_df.selectExpr("src as id").union(edges_df.selectExpr("dst as id")).distinct()
    vertices_df = vertices_df.withColumn("unique_id", UDF_HASH(vertices_df["id"]))

    pic = PowerIterationClustering(
        k=num_clusters,             # Number of clusters
        maxIter=max_iterations,     # Max iterations
        weightCol="weight"          # Use edge weights
    )

    # Assign clusters
    assignments = pic.assignClusters(new_edges_df)
    assignments.coalesce(1).write.csv(".", header=True, mode="overwrite")

    os.rename(glob.glob("part*.csv")[0], FILEPATH_VERTICES_CSV)
    # Example output:
    # id,cluster
    # -130306988,1
    # 112924765,0
    # 106229084,0
