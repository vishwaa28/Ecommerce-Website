from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()

class User(UserMixin, db.Model):
	__tablename__ = "users"
	id = db.Column(db.Integer, primary_key=True)
	name = db.Column(db.Text, nullable=False)
	email = db.Column(db.String(50), nullable=False)
	phone = db.Column(db.String(50), nullable=False)
	password = db.Column(db.String(250), nullable=False)
	admin = db.Column(db.Boolean, nullable=True, default=False)
	email_confirmed = db.Column(db.Boolean, nullable=True, default=False)
	cart = db.relationship('Cart', backref='buyer')
	orders = db.relationship("Order", backref='customer')

	def add_to_cart(self, itemid, quantity):
		item_to_add = Cart(itemid=itemid, uid=self.id, quantity=quantity)
		db.session.add(item_to_add)
		db.session.commit()

	def remove_from_cart(self, itemid, quantity):
		item_to_remove = Cart.query.filter_by(itemid=itemid, uid=self.id, quantity=quantity).first()
		db.session.delete(item_to_remove)
		db.session.commit()

class Item(db.Model):
	__tablename__ = "items"
	id = db.Column(db.Integer, primary_key=True)
	name = db.Column(db.String(100), nullable=False)
	price = db.Column(db.Float, nullable=False)
	category = db.Column(db.Text, nullable=False)
	image = db.Column(db.String(250), nullable=False)
	details = db.Column(db.String(250), nullable=False)
	price_id = db.Column(db.String(250), nullable=False)
	orders = db.relationship("Ordered_item", backref="item")
	in_cart = db.relationship("Cart", backref="item")
	reviews = db.relationship('Review', backref='item', cascade='all, delete-orphan')


class Cart(db.Model):
	__tablename__ = "cart"
	id = db.Column(db.Integer, primary_key=True)
	uid = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
	itemid = db.Column(db.Integer, db.ForeignKey('items.id'), nullable=False)
	quantity = db.Column(db.Integer, nullable=False, default=1)

class Order(db.Model):
	__tablename__ = "orders"
	id = db.Column(db.Integer, primary_key=True)
	uid = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
	date = db.Column(db.DateTime, nullable=False)
	status = db.Column(db.String(50), nullable=False)
	items = db.relationship("Ordered_item", backref="order")

class Ordered_item(db.Model):
	__tablename__ = "ordered_items"
	id = db.Column(db.Integer, primary_key=True)
	oid = db.Column(db.Integer, db.ForeignKey('orders.id'), nullable=False)
	itemid = db.Column(db.Integer, db.ForeignKey('items.id'), nullable=False)
	quantity = db.Column(db.Integer, db.ForeignKey('cart.quantity'), nullable=False)


class Review(db.Model):
	__tablename__ = "reviews"
	id = db.Column(db.Integer, primary_key=True)
	user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
	item_id = db.Column(db.Integer, db.ForeignKey('items.id'), nullable=False)
	review_text = db.Column(db.Text, nullable=False)
	rating = db.Column(db.Integer, nullable=False)
	created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
	user = db.relationship("User", backref="reviews")

	def add_review(user_id, item_id, review_text, rating):
		new_review = Review(user_id=user_id, item_id=item_id, review_text=review_text, rating=rating)
		db.session.add(new_review)
		db.session.commit()

	def get_reviews_for_product(item_id):
		return Review.query.filter_by(item_id=item_id).all()


