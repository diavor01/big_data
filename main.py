from functions import (write_titles_csv, 
                       write_edges_raw_csv, write_refs_csv,
                       write_edges_and_vertices_csv)

num_clusters = 5
max_iterations = 50

write_titles_csv("climate change")
write_refs_csv()
write_edges_raw_csv()
write_edges_and_vertices_csv(num_clusters, max_iterations)
