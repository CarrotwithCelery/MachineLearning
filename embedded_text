import os
import tkinter as tk
from tkinter import Entry, Button, Label, Scrollbar, ttk
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import subprocess


class DocumentSearch:
    folder_path = r"C:\Users\User\Documents\Embedded GPT"

    def __init__(self):
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.documents = []
        self.document_names = []
        self.load_documents()
        self.setup_gui()

    def load_documents(self):

        for filename in os.listdir(self.folder_path):
            if filename.endswith(".txt"):
                with open(os.path.join(self.folder_path, filename), 'r', encoding='utf-8') as file:
                    document_text = file.read()
                    self.documents.append(document_text)
                    self.document_names.append(filename)

    def encode_documents(self):
        self.document_embeddings = self.model.encode(self.documents)

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Document Search")

        query_label = Label(self.root, text="Enter your query:")
        query_label.pack()

        self.query_entry = Entry(self.root, width=50)
        self.query_entry.pack()

        self.results_tree = ttk.Treeview(self.root, columns=("Document Name", "Similarity Score"), show="headings")
        self.results_tree.heading("Document Name", text="Document Name")
        self.results_tree.heading("Similarity Score", text="Similarity Score")

        # Bind the open_file function to the double-click event on the table rows
        self.results_tree.bind("<Double-1>", self.open_file)

        vsb = Scrollbar(self.root, orient="vertical", command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=vsb.set)

        self.results_tree.pack(fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        search_button = Button(self.root, text="Search", command=self.perform_search)
        search_button.pack()

        reset_button = Button(self.root, text="Reset", command=self.reset_gui)
        reset_button.pack()

        show_chart_button = Button(self.root, text="Show Bar Chart", command=self.show_chart)
        show_chart_button.pack()

    def perform_search(self):
        query = self.query_entry.get()
        query_embedding = self.model.encode(query)

        similarities = util.pytorch_cos_sim(query_embedding, self.document_embeddings)[0]

        results = [(self.document_names[i], similarities[i]) for i in range(len(self.documents))]
        results.sort(key=lambda x: x[1], reverse=True)

        self.clear_results()
        self.display_results(results)
        self.display_bar_chart(results)

    def clear_results(self):
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

    def display_results(self, results):
        for doc_name, score in results:
            self.results_tree.insert("", "end", values=(doc_name, f"{score:.4f}"))

    def reset_gui(self):
        self.query_entry.delete(0, "end")
        self.clear_results()
        self.clear_bar_chart()

    def run(self):
        self.root.mainloop()

    def clear_bar_chart(self):
        # Clear the bar chart by removing the Matplotlib figure
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Canvas):
                widget.destroy()

    def show_chart(self):
        query = self.query_entry.get()
        query_embedding = self.model.encode(query)

        similarities = util.pytorch_cos_sim(query_embedding, self.document_embeddings)[0]

        results = [(self.document_names[i], similarities[i]) for i in range(len(self.documents))]
        results.sort(key=lambda x: x[1], reverse=True)

        self.display_bar_chart(results)

    def display_bar_chart(self, results):
        # Check if a graph is already displayed and destroy it
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Canvas):
                widget.destroy()

        # Extract document names and similarity scores
        doc_names, scores = zip(*results)

        # Determine bar colors based on similarity scores
        bar_colors = ['green' if score > 0.7 else 'skyblue' for score in scores]

        # Create a bar chart
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(doc_names, scores, color=bar_colors)
        ax.set_xlabel('Similarity Score')
        ax.set_title('Similarity Scores for Documents')

        # Add a legend for the color coding
        above_threshold = plt.Line2D([0], [0], color='green', lw=4, label='Score > 0.7')
        below_threshold = plt.Line2D([0], [0], color='skyblue', lw=4, label='Score <= 0.7')
        ax.legend(handles=[above_threshold, below_threshold])

        # Embed the Matplotlib figure in the tkinter GUI
        canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack()

        # Display the chart
        canvas.draw()

    def open_file(self, event=None):
        # Get the selected item from the results_tree
        item = self.results_tree.selection()[0]
        doc_name = self.results_tree.item(item, 'values')[0]

        # Construct the full file path
        file_path = os.path.join(self.folder_path, doc_name)

        # Use the 'start' command to open the file on Windows
        subprocess.Popen(['start', '', file_path], shell=True)

def main():
    document_search = DocumentSearch()
    document_search.encode_documents()
    document_search.run()


if __name__ == "__main__":
    main()
