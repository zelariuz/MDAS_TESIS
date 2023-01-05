#!/bin/sh

# shutdown api
sh stop_server.sh

# start api
sh start_server.sh

echo " twisted server restarted and running with PID $(paste *.pid)"