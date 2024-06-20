# Job Recommendation System

This project implements a job recommendation system using TF-IDF for text vectorization, SVD for dimensionality reduction, and FAISS for efficient similarity search. The system suggests job titles based on input skills.

## Prerequisites

Make sure to install the required packages before running the application:

```bash
pip install faiss-cpu pandas scikit-learn joblib
```

## File Structure

- `app.py`: Main script for running the job recommendation system.
- `tfidf_model.pkl`: Pre-trained TF-IDF model.
- `svd_model.pkl`: Pre-trained SVD model.
- `faiss_index.idx`: Pre-trained FAISS index.
- `jobs3.csv`: Dataset containing job titles.

## How to Run

1. **Load the Models and Data**:
    The script loads the pre-trained models and data files. Ensure the paths to these files are correctly specified:

    ```python
    tfidf_path = 'path/to/tfidf_model.pkl'
    svd_path = 'path/to/svd_model.pkl'
    faiss_path = 'path/to/faiss_index.idx'
    df_path = 'path/to/jobs3.csv'
    
    tfidf = joblib.load(tfidf_path)
    svd = joblib.load(svd_path)
    faiss_index = faiss.read_index(faiss_path)
    df = pd.read_csv(df_path)
    ```

2. **Get Recommendations**:
    The `get_recommendations` function takes the following parameters:
    - `index`: FAISS index
    - `tfidf`: TF-IDF model
    - `svd`: SVD model
    - `skills_input`: A string of input skills
    - `n_recommendations`: Number of recommendations to return (default is 10)

    Example usage:

    ```python
    skills_input = "machine learning, data analysis, python"
    recommendations = get_recommendations(faiss_index, tfidf, svd, skills_input)
    print(recommendations)
    ```

## Function Definition

### `get_recommendations`

```python
def get_recommendations(index, tfidf, svd, skills_input, n_recommendations=10):
    """
    Get job recommendations based on input skills.

    Parameters:
        index (faiss.Index): Pre-trained FAISS index.
        tfidf (TfidfVectorizer): Pre-trained TF-IDF model.
        svd (TruncatedSVD): Pre-trained SVD model.
        skills_input (str): Input skills as a string.
        n_recommendations (int): Number of recommendations to return.

    Returns:
        pd.Series: Recommended job titles.
    """
    skills_tfidf = tfidf.transform([skills_input])
    skills_reduced = svd.transform(skills_tfidf)

    distances, indices = index.search(skills_reduced, n_recommendations)
    
    return df['job_title'].iloc[indices[0]]
```

## Example

```python
# Load models and data
tfidf = joblib.load('path/to/tfidf_model.pkl')
svd = joblib.load('path/to/svd_model.pkl')
faiss_index = faiss.read_index('path/to/faiss_index.idx')
df = pd.read_csv('path/to/jobs3.csv')

# Input skills
skills_input = "machine learning, data analysis, python"

# Get recommendations
recommendations = get_recommendations(faiss_index, tfidf, svd, skills_input)
print(recommendations)
```

This should provide a list of job titles that best match the input skills based on the pre-trained models.

## Notes

- Ensure the paths to the models and dataset are correct.
- The `skills_input` should be a string containing the skills separated by commas.
- The number of recommendations returned can be adjusted by changing the `n_recommendations` parameter.

---
