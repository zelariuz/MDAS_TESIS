#!/bin/sh

if [ -f api.pid ];
then
 echo " Shutting down twisted server ..."
 PID_TWISTED="$(paste *.pid)"
 # PID_TWISTED="$(cat twistd.pid)"
 echo "pid: $PID_TWISTED"
 kill $PID_TWISTED
 echo "kill - SIGINT $PID_TWISTED"
 echo " Twisted server stopped."
 sleep 0.5
else
 echo " PID for process not found. Unnable to shut down twisted server."
fi
