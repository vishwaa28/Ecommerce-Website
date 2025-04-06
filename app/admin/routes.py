from _cffi_backend import string
from flask import Blueprint, render_template, url_for, flash, request
from werkzeug.utils import redirect
from ..db_models import Order, Item, db, Review
from ..admin.forms import AddItemForm, OrderEditForm
from ..funcs import admin_only
import pickle
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
import string
from collections import defaultdict
import calendar
from collections import Counter

admin = Blueprint("admin", __name__, url_prefix="/admin", static_folder="static", template_folder="templates")



fake_review_model = pickle.load(open("app/admin/ML_model.pkl", "rb"))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)


def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()


def analyze_review(review_text, rating):
    cleaned = preprocess_text(review_text)
    text_len = len(cleaned.split())
    capital_ratio = sum(1 for c in review_text if c.isupper()) / max(len(review_text), 1)
    punctuation_count = sum(1 for c in review_text if c in string.punctuation)
    has_exclamations = int('!' in review_text)

    embedding = get_bert_embedding(cleaned)
    extra_features = np.array([rating, text_len, capital_ratio, punctuation_count, has_exclamations])
    features = np.hstack((embedding, extra_features.reshape(1, -1)))

    is_fake = fake_review_model.predict(features)[0]

    if is_fake == 1:
        return "Fake", None
    else:
        sentiment = 1 if rating > 2 else 0
        return "Original", "Positive" if sentiment == 1 else "Negative"


@admin.route("/admin/reviews")
def admin_reviews():
    reviews = Review.query.all()

    analyzed_reviews = []

    for review in reviews:
        result, sentiment = analyze_review(review.review_text, review.rating)
        analyzed_reviews.append({
            "id": review.id,
            "user": review.user_id,
            "item": review.item_id,
            "text": review.review_text,
            "rating": review.rating,
            "created_at": review.created_at,
            "result": result,
            "sentiment": sentiment
        })

    return render_template("admin/admin_reviews.html", all_reviews=analyzed_reviews)


@admin.route("/")
def dashboard():
    reviews = Review.query.all()
    analyzed_reviews = []
    monthly_review_data = defaultdict(int)
    user_sentiment_counts = defaultdict(lambda: {'Positive': 0, 'Negative': 0})
    rating_counts = Counter([r.rating for r in reviews])
    rating_data = [rating_counts.get(i, 0) for i in range(1, 6)]  # 1★ to 5★

    for review in reviews:
        result, sentiment = analyze_review(review.review_text, review.rating)
        month_name = review.created_at.strftime('%b')  # 'Jan', 'Feb', etc.
        monthly_review_data[month_name] += 1
        if review.user is None:
            continue
        username = review.user.name  # using user.name from your User model
        chart_sentiment = review.sentiment if hasattr(review, 'sentiment') else 'Unknown'
        if chart_sentiment in ['Positive', 'Negative']:
            user_sentiment_counts[username][chart_sentiment] += 1
        analyzed_reviews.append({
            "id": review.id,
            "user": review.user_id,
            "item": review.item_id,
            "text": review.review_text,
            "rating": review.rating,
            "created_at": review.created_at,
            "result": result,
            "sentiment": sentiment
        })
    all_months = {month: monthly_review_data.get(month, 0) for month in calendar.month_abbr if month}
    # Extract lists for charting
    user_labels = list(user_sentiment_counts.keys())
    user_pos = [user_sentiment_counts[user]['Positive'] for user in user_labels]
    user_neg = [user_sentiment_counts[user]['Negative'] for user in user_labels]
    return render_template("admin/home.html", analyzed_reviews=analyzed_reviews,monthly_data=all_months,
                        user_labels=user_labels,
                       user_pos=user_pos,
                       user_neg=user_neg,
                           rating_data=rating_data
                           )

# @admin.route('/')
# @admin_only
# def dashboard():
#     orders = Order.query.all()
#     return render_template("admin/home.html", orders=orders)

@admin.route('/items')
@admin_only
def items():
    items = Item.query.all()
    return render_template("admin/items.html", items=items)

@admin.route('/add', methods=['POST', 'GET'])
@admin_only
def add():
    form = AddItemForm()

    if form.validate_on_submit():
        name = form.name.data
        price = form.price.data
        category = form.category.data
        details = form.details.data
        form.image.data.save('app/static/uploads/' + form.image.data.filename)
        image = url_for('static', filename=f'uploads/{form.image.data.filename}')
        price_id = form.price_id.data
        item = Item(name=name, price=price, category=category, details=details, image=image, price_id=price_id)
        db.session.add(item)
        db.session.commit()
        flash(f'{name} added successfully!','success')
        return redirect(url_for('admin.items'))
    return render_template("admin/add.html", form=form)

@admin.route('/edit/<string:type>/<int:id>', methods=['POST', 'GET'])
@admin_only
def edit(type, id):
    if type == "item":
        item = Item.query.get(id)
        form = AddItemForm(
            name = item.name,
            price = item.price,
            category = item.category,
            details = item.details,
            image = item.image,
            price_id = item.price_id,
        )
        if form.validate_on_submit():
            item.name = form.name.data
            item.price = form.price.data
            item.category = form.category.data
            item.details = form.details.data
            item.price_id = form.price_id.data
            form.image.data.save('app/static/uploads/' + form.image.data.filename)
            item.image = url_for('static', filename=f'uploads/{form.image.data.filename}')
            db.session.commit()
            return redirect(url_for('admin.items'))
    elif type == "order":
        order = Order.query.get(id)
        form = OrderEditForm(status = order.status)
        if form.validate_on_submit():
            order.status = form.status.data
            db.session.commit()
            return redirect(url_for('admin.dashboard'))
    return render_template('admin/add.html', form=form)

@admin.route('/delete/<int:id>')
@admin_only
def delete(id):
    to_delete = Item.query.get(id)
    for review in to_delete.reviews:
        db.session.delete(review)
    db.session.delete(to_delete)
    db.session.commit()
    flash(f'{to_delete.name} deleted successfully', 'error')
    return redirect(url_for('admin.items'))

@admin.route('/delete_review/<int:id>')
@admin_only
def delete_review(id):
    to_delete = Review.query.get_or_404(id)
    db.session.delete(to_delete)
    db.session.commit()
    flash(f'Review "{to_delete.review_text}" deleted successfully', 'error')
    return redirect(url_for('admin.admin_reviews'))



