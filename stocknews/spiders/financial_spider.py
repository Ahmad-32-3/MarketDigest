import scrapy
from stocknews.items import StocknewsItem

class FinancialSpider(scrapy.Spider):
    name = 'financial'
    allowed_domains = ['finance.yahoo.com']
    start_urls = ['https://finance.yahoo.com/news/']

    def parse(self, response):
        articles = response.css('li.stream-item')


        self.logger.info(f"Found {len(articles)} articles on the page")

        for article in articles:
            item = StocknewsItem()
            item['title'] = article.css('h3.yf-1y7058a::text').get()
            item['url'] = response.urljoin(article.css('a::attr(href)').get())
            item['source'] = 'Yahoo Finance'

            yield scrapy.Request(item['url'], callback=self.parse_article, meta={'item': item})

    def parse_article(self, response):
        item = response.meta['item']
        item['content'] = ' '.join(response.css('p::text').getall())
        item['date'] = response.css('time::attr(datetime)').get()

        yield item
