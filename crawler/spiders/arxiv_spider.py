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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_items = 5
        self.item_count = 0

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        sources = crawler.settings.get('sources', [])
        if sources:
            spider.max_items = sources[0].get('max_items', 5)
        else:
            spider.max_items = 5
        spider.logger.info(f"Spider initialized with max_items: {spider.max_items}")
        return spider
    
    def start_requests(self):
        sources = self.settings.get('sources', [])
        if sources:
            urls = sources[0].get('start_urls', self.start_urls)
        else:
            urls = self.start_urls
        
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        self.logger.info(f"Starting parse. Current count: {self.item_count}, Max: {self.max_items}")
        
        dts = response.css('dl > dt')
        dds = response.css('dl > dd')
        
        for dt, dd in zip(dts, dds):
            # Check limit BEFORE processing each item
            if self.item_count >= self.max_items:
                self.logger.info(f"Reached max items limit: {self.max_items}. Stopping.")
                return
                
            title_block = dd.css('div.list-title.mathjax::text').getall()
            title = " ".join(t.strip() for t in title_block if t.strip()).replace('Title:','',1).strip()
            
            if not title:
                continue
                
            authors = [a.strip() for a in dd.css('div.list-authors a::text').getall()]
            abstract = " ".join(dd.css('p.mathjax::text').getall()).strip()
            
            base_item = {
                "title": title,
                "authors": authors,
                "abstract": abstract,
            }
            
            # Increment counter immediately when we decide to yield an item
            self.item_count += 1
            self.logger.info(f"Processing item {self.item_count}/{self.max_items}: {title[:50]}...")
            
            # For simplicity, just yield the base item without PDF download
            # If you want PDFs, uncomment the PDF logic below
            yield base_item
            
            # PDF download logic (optional - comment out for faster testing)
            # pdf_rel = dt.css('span.list-identifier a[title="Download PDF"]::attr(href)').get()
            # if pdf_rel:
            #     pdf_url = response.urljoin(pdf_rel)
            #     yield scrapy.Request(pdf_url, callback=self.save_pdf, meta={"base_item": base_item})
            # else:
            #     yield base_item

    def save_pdf(self, response):
        base_item = response.meta['base_item']
        arxiv_id = response.url.rstrip('/').split('/')[-1]
        pdf_dir = "data/pdfs"
        os.makedirs(pdf_dir, exist_ok=True)
        path = os.path.join(pdf_dir, f"{arxiv_id}.pdf")
        with open(path, 'wb') as f:
            f.write(response.body)
        base_item['file_path'] = path
        yield base_item