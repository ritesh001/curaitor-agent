import scrapy
import os
import logging

class ArxivSpider(scrapy.Spider):
    name = "arxiv"
    allowed_domains = ["arxiv.org"]
    start_urls = ["https://arxiv.org/list/physics/new"]
    custom_settings = {
        'DOWNLOAD_DELAY': 1,
        'CONCURRENT_REQUESTS': 4,
    }

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     # Read from settings (passed from config)
    #     sources = self.settings.get('sources', [])
    #     if sources:
    #         self.max_items = sources[0].get('max_items', 5)
    #     else:
    #         self.max_items = 5
    #     self.item_count = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Default max_items, will be overridden in from_crawler
        self.max_items = 5
        self.item_count = 0

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        # Now we can access settings
        sources = crawler.settings.get('sources', [])
        if sources:
            spider.max_items = sources[0].get('max_items', 5)
        else:
            spider.max_items = 5
        return spider
    
    def start_requests(self):
        urls = self.settings.get('sources')[0]['start_urls']
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    # def parse(self, response): ## does not work
    #     for entry in response.css('div.listing .list-title'):
    #         pdf_url = entry.xpath('../a[@title="Download PDF"]/attribute::href').get()
    #         yield response.follow(pdf_url, self.parse_pdf)

    # def parse(self, response): ## at least crawled.json is not empty
    #     self.logger.info("Parsing %s", response.url)
    #     papers = response.css('dl > dt')
    #     if not papers:
    #         self.logger.warning("No <dt> nodes found; selector may be wrong")
    #     for dd in response.css('dl > dd'):
    #         title_block = dd.css('div.list-title.mathjax::text').getall()
    #         title = " ".join(t.strip() for t in title_block if t.strip()).replace('Title:','',1).strip()
    #         authors = [a.strip() for a in dd.css('div.list-authors a::text').getall()]
    #         abstract = " ".join(dd.css('p.mathjax::text').getall()).strip()
    #         if not title:
    #             continue
    #         yield {
    #             "title": title,
    #             "authors": authors,
    #             "abstract": abstract,
    #         }

    def parse(self, response):
        dts = response.css('dl > dt')
        dds = response.css('dl > dd')
        for dt, dd in zip(dts, dds):
            pdf_rel = dt.css('span.list-identifier a[title="Download PDF"]::attr(href)').get()
            title_block = dd.css('div.list-title.mathjax::text').getall()
            title = " ".join(t.strip() for t in title_block if t.strip()).replace('Title:','',1).strip()
            authors = [a.strip() for a in dd.css('div.list-authors a::text').getall()]
            abstract = " ".join(dd.css('p.mathjax::text').getall()).strip()
            base_item = {
                "title": title,
                "authors": authors,
                "abstract": abstract,
            }
            if pdf_rel:
                pdf_url = response.urljoin(pdf_rel)
                yield scrapy.Request(pdf_url, callback=self.save_pdf, meta={"base_item": base_item})
            else:
                yield base_item

    def save_pdf(self, response):
        base_item = response.meta['base_item']
        # Build deterministic filename from arXiv ID in URL
        arxiv_id = response.url.rstrip('/').split('/')[-1]  # e.g. 2401.12345
        pdf_dir = "data/pdfs"
        os.makedirs(pdf_dir, exist_ok=True)
        path = os.path.join(pdf_dir, f"{arxiv_id}.pdf")
        with open(path, 'wb') as f:
            f.write(response.body)
        base_item['file_path'] = path
        yield base_item

    def parse_pdf(self, response):
        path = self.settings.get('output')['path'].replace('.json', '') + '_tmp.pdf'
        with open(path, 'wb') as f:
            f.write(response.body)
        yield {'file_path': path}