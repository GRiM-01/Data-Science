from urllib.parse import urlparse

url_string = input('Link: ')

parsed_url = urlparse(url_string)

# Extract components
scheme = parsed_url.scheme
netloc = parsed_url.netloc
path = parsed_url.path

print("Scheme:", scheme)
print("Netloc:", netloc)
print("Path:", path)
