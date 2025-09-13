from scrapy.crawler import CrawlerProcess
import yaml
from crawler.spiders.arxiv_spider import ArxivSpider

# def run_crawler():
#     with open('config.yaml') as f:
#         settings = yaml.safe_load(f)
#     process = CrawlerProcess(settings={**settings, 'FEED_FORMAT': 'json', 'FEED_URI': 'crawled.json'})
#     # process.crawl('arxiv')
#     process.crawl(ArxivSpider)
#     process.start()

def run_crawler():
    with open('config.yaml') as f:
        # settings = yaml.safe_load(f)
        config = yaml.safe_load(f)

    max_items = config.get('sources', [{}])[0].get('max_items', 5)
    
    feeds = {
        'crawled.json': {
            'format': 'json',      # emits a single JSON array
            'encoding': 'utf8',
            'overwrite': True,     # prevent appending multiple JSON docs
            'indent': 2,
        }
    }

    # process = CrawlerProcess(settings={**settings, 'FEEDS': feeds})
        # Pass config as settings so spider can access it
    process = CrawlerProcess(settings={
        **feeds, 
        'FEEDS': feeds,
        'sources': config.get('sources', []),
        'CLOSESPIDER_ITEMCOUNT': max_items,  # Built-in Scrapy limit
    })
    process.crawl(ArxivSpider)
    process.start()