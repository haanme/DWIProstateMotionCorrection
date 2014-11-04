#!/bin/sh

# Kill main script processes
ids=$(ps -A | grep runall_ | awk '{print $1}')
for id in $ids
    echo kill $id
    kill $id
done
# Kill currently running subprocesses
ids=$(ps -A | grep dlib_ | awk '{print $1}')
for id in $ids
    echo kill $id
    kill $id
done

