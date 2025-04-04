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

def analyze_review(review):
    review_text = review.review_text
    rating = review.rating
    cleaned = preprocess_text(review_text)
    text_len = len(cleaned.split())
    capital_ratio = sum(1 for c in review_text if c.isupper()) / max(len(review_text), 1)
    punctuation_count = sum(1 for c in review_text if c in string.punctuation)
    has_exclamations = int('!' in review_text)

    embedding = get_bert_embedding(cleaned)
    extra_features = np.array([rating, text_len, capital_ratio, punctuation_count, has_exclamations])
    features = np.concatenate((embedding.flatten(), extra_features)).reshape(1, -1)

    is_fake = fake_review_model.predict(features)[0]
    return "Fake" if is_fake == 1 else "Original"

@admin.route("/admin/reviews")
# @admin_only  # Uncomment this line if admin_only decorator is defined
def admin_reviews():
    reviews = Review.query.all()
    return render_template("admin/admin_reviews.html", reviews=reviews)

@admin.route("/")
# @admin_only  # Uncomment this line if admin_only decorator is defined
def dashboard():
    reviews = Review.query.all()
    analyzed_reviews = []
    for review in reviews:
        result = analyze_review(review)
        analyzed_reviews.append(result)
    return render_template("admin/home.html", reviews=analyzed_reviews)

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
    db.session.delete(to_delete)
    db.session.commit()
    flash(f'{to_delete.name} deleted successfully', 'error')
    return redirect(url_for('admin.items'))


