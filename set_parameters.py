from flask_wtf import Form
from wtforms import StringField, PasswordField, SubmitField,IntegerField
from wtforms.widgets import html5
from wtforms.validators import DataRequired, Email, Length

class SetParameters(Form):
  """first_name = StringField('First name', validators=[DataRequired("Please enter your first name.")])
  last_name = StringField('Last name', validators=[DataRequired("Please enter your last name.")])
  email = StringField('Email', validators=[DataRequired("Please enter your email address."), Email("Please enter your email address.")])
  password = PasswordField('Password', validators=[DataRequired("Please enter a password."), Length(min=6, message="Passwords must be 6 characters or more.")])"""


  no_of_topic=IntegerField("Topic Number:",widget=html5.NumberInput(),validators=[DataRequired("Please enter number of topic u desired.")])
  no_of_words=IntegerField("Top Words Number:",widget=html5.NumberInput(),validators=[DataRequired("Please enter number of words u desired.")])
  no_of_documents=IntegerField("Top document Number:",widget=html5.NumberInput(),validators=[DataRequired("Please enter number of documents u desired.")])
  submit = SubmitField('Show Results')
