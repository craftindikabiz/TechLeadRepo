import pandas as pd
import numpy as np
from transformer import pipeline

# Load dataset
data = {
    'Title': ['To Kill a Mockingbird', '1984', 'Moby Dick', 'The Great Gatsby', 'War and Peace'],
    'Author': ['Harper Lee', 'George Orwell', 'Herman Melville', 'F. Scott Fitzgerald', 'Leo Tolstoy'],
    'Genre': ['Fiction', 'Dystopian', 'Adventure', 'Classic', 'Historical']
}
df = pd.DataFrame(data)

# Missing part: Display the first few rows of the dataset


# Missing part: Calculate the length of each book title and add it as a new column


# Missing part: Calculate the average title length

print(f'Average Title Length: {average_title_length}')

# Load a pre-trained language model for text generation
generator = pipeline('text-generation', model='gpt2')

# Missing part: Generate a short description for a given book title
book_title = "To Kill a Mockingbird"
generated_text = generator
print(f'Generated Description for "{book_title}": {generated_text[0]["generated_text"]}')
