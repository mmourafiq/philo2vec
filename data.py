import os
from os import listdir
from os.path import isfile, join

import json
from urllib.request import urlopen, Request
from lxml import html


class PlatoData(object):
    BASE_URL = 'http://plato.stanford.edu/'
    USER_AGENT = 'Mozilla/5.0 (compatible; MSIE 8.0; Windows NT 6.0)'
    DATA_DIR = './data'

    def _get_page(self, url):
        request = Request(url, headers={'User-Agent': self.USER_AGENT})
        response = urlopen(request)
        html = response.read()
        response.close()
        return html

    def get_index(self):
        url = '{}{}'.format(self.BASE_URL, 'contents.html')
        response = self._get_page(url)
        response = html.fromstring(response)
        return response.xpath('//div[@id="content"]//li/a/@href')

    def _get_entry(self, entry):
        url = '{}{}'.format(self.BASE_URL, entry)
        entry_filename = '{}/{}'.format(self.DATA_DIR, entry.split('/')[1])
        if os.path.isfile(entry_filename):
            return
        response = self._get_page(url)
        response = html.fromstring(response)
        print(response.xpath('//div[@id="aueditable"]//h1//text()'))
        data = response.xpath('//div[@id="main-text"]//p//text()')
        with open(entry_filename, 'w') as outfile:
            json.dump(data, outfile)
        outfile.close()

    def get_entries(self):
        for entry in self.get_index():
            self._get_entry(entry)

    def prepare_corpus(self, download=False):
        if download:
            self.get_entries()

        def _get_entry_text(entry):
            with open('{}/{}'.format(self.DATA_DIR, entry), 'r') as f:
                data = json.loads(f.read())
                return ''.join(''.join(data).split('\n'))

        entries = []
        for f in listdir(self.DATA_DIR):
            if isfile(join(self.DATA_DIR, f)) and not f.startswith('.'):
                entries.append(_get_entry_text(f))
        entries = '. '.join(entries)

        with open('{}/data'.format(self.DATA_DIR), 'w') as outfile:
            outfile.write(entries)


