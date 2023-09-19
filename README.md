# Document Search GUI with Similarity Scoring

This Python application provides a Graphical User Interface (GUI) for searching and retrieving text documents based on similarity scores. It utilizes the Sentence Transformers library to calculate document embeddings and cosine similarities for efficient document retrieval. Additionally, it displays search results in a table and visualizes similarity scores using a bar chart.

## How It Works

1. **Document Loading**: The application loads text documents from a specified folder path. Each document in the folder should be a `.txt` file. You can set the folder path in the `folder_path` variable at the beginning of the script.

2. **Document Embeddings**: It uses a pre-trained Sentence Transformers model (default: 'paraphrase-MiniLM-L6-v2') to encode the loaded documents into vector embeddings. The embeddings are used to calculate similarity scores.

3. **Graphical User Interface (GUI)**: The GUI allows users to perform document searches using a query. It includes the following components:

   - Input field for entering a search query.
   - A table (Treeview widget) for displaying search results, including document names and similarity scores.
   - "Search" button to trigger the search.
   - "Reset" button to clear the input field and search results.
   - "Show Bar Chart" button to display a bar chart of similarity scores for the search results.

4. **Search and Results**: When the user enters a query and clicks the "Search" button or presses Enter, the application calculates the similarity scores between the query and all documents. It displays the search results sorted by similarity score in descending order. The table includes document names and similarity scores.

5. **Bar Chart Visualization**: Clicking the "Show Bar Chart" button displays a bar chart visualizing the similarity scores. Documents with scores above a threshold (default: 0.7) are highlighted in green, while those below the threshold are displayed in blue.

6. **Document Opening**: You can double-click on a row in the search results table to open the corresponding document file using the default system viewer.

## How to Use

1. Ensure you have the necessary libraries installed. You can install them using pip:

   ```bash
   pip install sentence-transformers tkinter matplotlib
