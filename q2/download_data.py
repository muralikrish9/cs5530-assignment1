import requests, re

r = requests.get('https://app.box.com/s/ji910ez3ycw137rw07xnhielxey7ww41')
m = re.search(r'"typedID":"f_(\d+)"', r.text)
if m:
    fid = m.group(1)
    print(f'File ID: {fid}')
    dl = requests.get(
        f'https://app.box.com/index.php?rm=box_download_shared_file&shared_name=ji910ez3ycw137rw07xnhielxey7ww41&file_id=f_{fid}',
        allow_redirects=True
    )
    print(f'Status: {dl.status_code}, Size: {len(dl.content)}')
    with open('student_performance.csv', 'wb') as f:
        f.write(dl.content)
    print('Saved')
    print(dl.content[:300].decode('utf-8', errors='replace'))
else:
    print('No file ID found')
