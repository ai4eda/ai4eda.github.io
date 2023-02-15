serve:
	bundle exec jekyll serve
build:
	bundle exec jekyll build && echo `date '+%Y-%m-%d'` > ./_includes/last-updated.txt
