#REST webservice
import json
import os
import shutil
from socket import gethostname
import webbrowser

import cherrypy
from cherrypy.lib.static import serve_file
from cherrypy import expose
import numpy as np

from ..workflows import get_workflows, get_workflow
from scripts.form_scripts import get_form

MEDIA_DIR = os.path.join(os.path.dirname(__file__), 'scripts')
FILE_DIR = os.path.join(os.getcwd(), 'files')
if 'bips' in gethostname():
    url_prefix = '\/\/bips.incf.org:8080\/'
else:
    url_prefix = ''

class MyEncoder(json.JSONEncoder):
    def default(self, o):
        """Implement this method in a subclass such that it returns
        a serializable object for ``o``, or calls the base implementation
        (to raise a ``TypeError``).

        For example, to support arbitrary iterators, you could
        implement default like this::

            def default(self, o):
                try:
                    iterable = iter(o)
                except TypeError:
                    pass
                else:
                    return list(iterable)
                return JSONEncoder.default(self, o)

        """
        try:
            return super(MyEncoder, self).default(o)
        except TypeError:
            return ""

class BIPS(object):
    def __init__(self, *args, **kwargs):
        tags = []
        mapper = {}
        workflows = {}
        for wf, value in get_workflows():
            wf_tags = sorted(np.unique([tag.lower() for tag in value['object'].tags]).tolist())
            for tag in wf_tags:
                if tag not in mapper:
                    mapper[tag] = []
                mapper[tag].append(wf)
            tags.extend(wf_tags)
            workflows[wf] = value['object'].help.split('\n')[1]
        self.tags_ = mapper.keys()
        self.mapper_ = mapper
        self.wf_ = workflows

    @expose
    def index(self):
        with open(os.path.join(MEDIA_DIR, 'index.html')) as fp:
            msg = fp.readlines()
        return msg

    @expose
    def workflows(self, tags=None):
        with open(os.path.join(MEDIA_DIR, 'workflows.html')) as fp:
            msg = fp.readlines()
        return msg

    @expose
    def edit_config(self,uuid='7757e3168af611e1b9d5001e4fb1404c'):
        wf = get_workflow(uuid)
        try:
            view = wf.html_view()
        except:
            view = None
        conf = wf.config_ui()
        with open(os.path.join(MEDIA_DIR, 'edit_config.html')) as fp:
            m = fp.readlines()
            form = get_form(conf,view)
            
            msg = '\n'.join(m).replace('**TEMPLATE**',form)
        return msg


    @expose
    def queryworkflows(self, tags=None):
        print tags
        cherrypy.response.headers['Content-Type'] = 'application/json'
        if tags:
            wfs = []
            for tag in tags.split():
                wfs.extend(self.mapper_[tag])
            return json.dumps([{'uuid': wf, 'desc': self.wf_[wf]} for wf in wfs])
        else:
            return json.dumps([{'uuid': uuid, 'desc': desc} for uuid, desc in self.wf_.items()])

    @expose
    def tags(self, query):
        tags = self.tags_
        if query:
            query = query.split()
            if len(query):
                pre = ' '.join(query[:-1])
                if pre:
                    pre += ' '
                query = query[-1]
            else:
                query = ' '
                pre=''
            tags = [pre + tag for tag in self.tags_ if query.lower() in tag]
        cherrypy.response.headers['Content-Type'] = 'application/json'
        return json.dumps(tags)

    """
        msg = ["<h2>Welcome to BIPS</h2>"]
        msg.append('<ul>')
        for wf, value in get_workflows():
            msg += [('<li><a href="info?uuid=%s">%s</a> <a href="configure'
                     '?uuid=%s">Configure</a> %s</li>') % (wf, wf, wf,
                                        value['object'].help.split('\n')[1])]
        msg.append('</li>')
        return '\n'.join(msg)
    """

    @expose
    def info(self, uuid):
        wf = get_workflow(uuid)
        val = wf.get()
        json_str =  val #json.dumps(val, cls=MyEncoder)
        config_str = wf.config_ui().get() #json.dumps(wf.config_ui().get(), cls=MyEncoder)
        img_file = ''
        cherrypy.response.headers['Content-Type'] = 'application/json'
        return json.dumps({'jsonconfig': json_str, 'workflowconfig': config_str},
                          cls=MyEncoder)
    '''
            msg = """

    var str = JSON.stringify(%s, null, 2);
    $(this).append('<h3>Workflow info</h3>');
    console.log($(this));
    /*
    document.write('<pre>' + syntaxHighlight(str) +'</pre>');
    var str2 = JSON.stringify(%s, null, 2);
    document.write('<h3>Workflow config</h3>')
    document.write('<pre>' + syntaxHighlight(str2) +'</pre>');
    document.write('<h3>Workflow graph</h3>')
    document.write('<img src="%s" />')
    */
    </script>
    """ % (json_str, config_str, img_file)
            return msg
    '''
    @expose
    def configure(self, uuid):
        wf = get_workflow(uuid)
        val = wf.get()

    @expose
    def demo(self):
        with open(os.path.join(MEDIA_DIR, 'demo.html')) as fp:
            msg = fp.readlines()
        return msg

    @expose
    def uploadhandler(self, **kwargs):
        if 'files[]' not in kwargs:
            return
        myFile = kwargs['files[]']
        outfile = os.path.join(FILE_DIR, myFile.filename)
        with open(outfile, 'wb') as fp:
            shutil.copyfileobj(myFile.file, fp)
        cherrypy.log('Saved file: %s' % outfile)
        if os.path.isfile(outfile):
            print "getting info: %s" % outfile
            size = os.path.getsize(outfile)
            from nibabel import load
            import Image
            import numpy as np
            data = load(outfile).get_data()
            if len(data.shape) == 4:
                slice = data[data.shape[0]/2,:,:,0]
            else:
                slice = data[data.shape[0]/2,:,:]
            print slice.shape
            im = Image.fromarray(np.squeeze(255.*slice/np.max(np.abs(slice))).astype(np.uint8))
            outpng = outfile+'.png'
            im.save(outpng)
            out = {"name": myFile.filename,
                   "size": size,
                   "url": "%sfiles\/%s" % (url_prefix, myFile.filename),
                   "thumbnail_url":"%sthumbnails\/%s.png" % (url_prefix, myFile.filename),
                   "delete_url": "%sdeletehandler?file=%s" % (url_prefix, myFile.filename),
                   "delete_type": "DELETE"
            }
        else:
            out = {}
        cherrypy.response.headers['Content-Type'] = 'application/json'
        return json.dumps([out])

    @expose
    def deletehandler(self, file):
        outfile = os.path.join(FILE_DIR, file)
        if os.path.isfile(outfile):
            cherrypy.log('Deleting file: %s' % outfile)
            os.unlink(outfile)
            os.unlink(outfile+'.png')

def open_page():
    #pass
    webbrowser.open("http://127.0.0.1:8080")

def start_service():
    #configure ip address and port for web service
    if not os.path.exists(FILE_DIR):
        os.mkdir(FILE_DIR)
    config = {'/': {'tools.staticdir.on': True,
                    'tools.staticdir.dir': os.getcwd()},
              '/css': {'tools.staticdir.on': True,
                       'tools.staticdir.dir': os.path.join(MEDIA_DIR, 'css')},
              '/js': {'tools.staticdir.on': True,
                      'tools.staticdir.dir': os.path.join(MEDIA_DIR, 'js')},
              '/cors': {'tools.staticdir.on': True,
                        'tools.staticdir.dir': os.path.join(MEDIA_DIR, 'cors')},
              '/img': {'tools.staticdir.on': True,
                       'tools.staticdir.dir': os.path.join(MEDIA_DIR, 'img')},
              '/thumbnails': {'tools.staticdir.on': True,
                       'tools.staticdir.dir': FILE_DIR},
              '/files': {'tools.staticdir.on': True,
                         'tools.staticdir.dir': FILE_DIR},
              }
    #start webservice
    certfile = os.path.join(os.environ['HOME'], 'certinfo')
    if os.path.exists(certfile):
        cherrypy.log('Loading cert info: %s' % certfile)
        cherrypy.config.update(json.load(open(certfile)))
    else:
        cherrypy.log('Cert info unavailable')
    cherrypy.engine.subscribe('start', open_page)
    cherrypy.tree.mount(BIPS(), '/', config=config)
    cherrypy.engine.start()
    cherrypy.engine.block()
    #cherrypy.quickstart(BIPS())
