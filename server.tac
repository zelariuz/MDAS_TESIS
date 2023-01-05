#!/usr/bin/python
# -*- coding: utf-8 -*-

import os, sys
cwd = os.getcwd()
# needed to import local directory modules
sys.path.append(cwd)

from twisted.web import server
from twisted.application import internet, service
from server_factory import ServerFactory

# init app
application = service.Application("Back MDAS TAPP")
root = ServerFactory()
site = server.Site(root)
site_port = 7777

# launch server
server = internet.TCPServer(site_port, site) #int(configdict.get("GENERAL","port"))
server.setServiceParent(application)