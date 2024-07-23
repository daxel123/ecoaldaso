from flet import app, Page
from ui import main_page

def main(page: Page):
    # Set the size of the window


    # Call your main_page function
    main_page(page)

app(target=main)
