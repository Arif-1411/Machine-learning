from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load artifacts
model      = pickle.load(open('artifacts/model.pkl',          'rb'))
books_name = pickle.load(open('artifacts/books_name.pkl',     'rb'))
book_pivot = pickle.load(open('artifacts/pivot_df.pkl',       'rb'))

# final_rating_df is optional (for images)
try:
    final_rating = pickle.load(open('artifacts/final_rating_df.pkl', 'rb'))
    HAS_IMAGES = True
except Exception:
    final_rating = None
    HAS_IMAGES = False


def recommend_book(book_name):
    book_id = np.where(book_pivot.index == book_name)[0][0]
    _, suggestion = model.kneighbors(
        book_pivot.iloc[book_id, :].values.reshape(1, -1),
        n_neighbors=6
    )

    results = []
    for idx in suggestion[0][1:]:  # skip index 0 (same book)
        name = book_pivot.index[idx]
        image = None
        if HAS_IMAGES and final_rating is not None:
            row = final_rating[final_rating['Title'] == name]
            if not row.empty:
                image = row.iloc[0]['image_url']
        results.append({'title': name, 'image': image})
    return results


@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    selected_book   = None

    if request.method == 'POST':
        selected_book   = request.form.get('book_name')
        recommendations = recommend_book(selected_book)

    return render_template('index.html',
                           books_name=books_name,
                           recommendations=recommendations,
                           selected_book=selected_book)


if __name__ == '__main__':
    app.run(debug=True)