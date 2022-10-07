#! /bin/bash


check_envs () {
    DOCKER_CUSTOM_USER_OK=true;
    if [ -z ${DOCKER_USER_NAME+x} ]; then 
        DOCKER_CUSTOM_USER_OK=false;
        return;
    fi
    if [ -z ${DOCKER_USER_ID+x} ]; then 
        DOCKER_CUSTOM_USER_OK=false;
        return;
    else
        if ! [ -z "${DOCKER_USER_ID##[0-9]*}" ]; then 
            echo -e "\033[1;33mWarning: User-ID should be a number. Falling back to defaults.\033[0m"
            DOCKER_CUSTOM_USER_OK=false;
            return;
        fi
    fi
    if [ -z ${DOCKER_USER_GROUP_NAME+x} ]; then 
        DOCKER_CUSTOM_USER_OK=false;
        return;
    fi
    if [ -z ${DOCKER_USER_GROUP_ID+x} ]; then 
        DOCKER_CUSTOM_USER_OK=false;
        return;
    else
        if ! [ -z "${DOCKER_USER_GROUP_ID##[0-9]*}" ]; then 
            echo -e "\033[1;33mWarning: Group-ID should be a number. Falling back to defaults.\033[0m"
            DOCKER_CUSTOM_USER_OK=false;
            return;
        fi
    fi
}


setup_env_user () {
    USER=$1
    USER_ID=$2
    GROUP=$3
    GROUP_ID=$4

    useradd -m $USER

    ## Copy bash configs
    cp /root/.profile /home/$USER/
    cp /root/.bashrc /home/$USER/
    
    ## Copy terminator configs
    mkdir -p /home/$USER/.config/terminator
    cp /config /home/$USER/.config/terminator/config
    mkdir -p /root/.config/terminator
    cp /config /root/.config/terminator/config
    
    # Copy SSH keys & fix owner
    if [ -d "/root/.ssh" ]; then
        cp -rf /root/.ssh /home/$USER/
        chown -R $USER:$GROUP /home/$USER/.ssh
    fi
    
    ## Fix owner
    chown $USER:$GROUP /home/$USER
    chown -R $USER:$GROUP /home/$USER/.config
    chown $USER:$GROUP /home/$USER/.profile
    chown $USER:$GROUP /home/$USER/.bashrc

    ## This a trick to keep the evnironmental variables of root which is important!
    echo "if ! [ \"$DOCKER_USER_NAME\" = \"$(id -un)\" ]; then" >> /root/.bashrc
    echo "    cd /home/$DOCKER_USER_NAME" >> /root/.bashrc
    echo "    su $DOCKER_USER_NAME" >> /root/.bashrc
    echo "fi" >> /root/.bashrc

    ## Setup Password-file
    PASSWDCONTENTS=$(grep -v "^${USER}:" /etc/passwd)
    GROUPCONTENTS=$(grep -v -e "^${GROUP}:" -e "^docker:" /etc/group)
    (echo "${PASSWDCONTENTS}" && echo "${USER}:x:$USER_ID:$GROUP_ID::/home/$USER:/bin/bash") > /etc/passwd
    (echo "${GROUPCONTENTS}" && echo "${GROUP}:x:${GROUP_ID}:") > /etc/group
    (if test -f /etc/sudoers ; then echo "${USER}  ALL=(ALL)   NOPASSWD: ALL" >> /etc/sudoers ; fi)
}


# Solve authority issues
setup_specific_user_setting () {
    sudo chmod 777 /root
    
    # ROS
    echo "source /home/$USER/catkin_ws/devel/setup.bash" >> /root/.bashrc
    sudo mkdir /home/$USER/.ros/
    sudo chmod -R 777 /home/$USER/.ros/

    # pygame
    sudo chmod -R 777 /home/$USER/.config/
    sudo chmod -R 777 /dev/input

    # Isaac
    sudo cp -r /root/isaacgym/ /home/$USER/isaacgym

    # isaac gym
    echo "export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}" >> /home/$USER/.bashrc
}


#---------- main ----------#


# Create new user
check_envs

# Setup Environment
echo "DOCKER_USER Input is set to '$DOCKER_USER_NAME:$DOCKER_USER_ID:$DOCKER_USER_GROUP_NAME:$DOCKER_USER_GROUP_ID'";
echo "Setting up environment for user=$DOCKER_USER_NAME"
setup_env_user $DOCKER_USER_NAME $DOCKER_USER_ID $DOCKER_USER_GROUP_NAME $DOCKER_USER_GROUP_ID

setup_specific_user_setting

# Run CMD from Docker
"$@"
