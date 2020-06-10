class infographicInfo:
    def __init__(self,currow):
        self.views = int(currow['views'])
        self.likes = int(currow['likes'])
        self.shares = int(currow['shares'])
        self.comments = int(currow['comments'])
        self.staffPick = currow['staffPick']
        self.imgurl = currow['url_img_ful_res']
        self.mainurl = currow['url']
        self.title = currow['igTitle']
        self.category = currow['category']
        self.description = currow['description']
        self.title = currow['title']
        self.tags = currow['tags']