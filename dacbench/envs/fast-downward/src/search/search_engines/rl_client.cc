#include "rl_client.h"
#include <iostream>

namespace rl_client {

RLClient::RLClient(int port, std::string ip_address) : port(port), ip_address(ip_address), sock(0) {}

bool RLClient::init_connection() {

    printf("\nPort: %i\n", port);
    
    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        printf("\n Socket creation error \n");
        return false;
    }

    memset(&serv_addr, '0', sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);

    // Convert IPv4 and IPv6 addresses from text to binary form 
    if(inet_pton(AF_INET, ip_address.c_str(), &serv_addr.sin_addr)<=0)
    {
        printf("\nInvalid address/ Address not supported \n");
        return false;
    }

    sockfd = connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr));

    if (sockfd < 0)
    {
        printf("\nConnection Failed \n");
        return false;
    }
    return true;
}

void RLClient::send_msg(const std::string& msg) const {
    char* cstr_msg = new char[msg.size() + 1];
    msg.copy(cstr_msg, msg.size() + 1);
    cstr_msg[msg.size()] = '\0'; 
    send(sock , cstr_msg, strlen(cstr_msg), 0);
    delete[] cstr_msg;
}

void RLClient::send_msg(const std::map<int, std::map<std::string, double>>& open_lists_stats, double reward, bool done) const {
    std::string py_dict = "{";
    for (auto& config_values : open_lists_stats) {
        int config = config_values.first;
        py_dict += "\"" + std::to_string(config) + "\" : {";

        for (auto& value_pair : config_values.second) {
            py_dict += "\"" + value_pair.first + "\" : " + std::to_string(value_pair.second) + ",";
        }
        py_dict = py_dict.substr(0, py_dict.size() - 1);
        py_dict += "},";
    }
    py_dict += "\"reward\" : " + std::to_string(reward);
    py_dict += ",\"done\" : ";
    py_dict += done ? std::to_string((double)1) : std::to_string((double)0);
    py_dict += "}";
    std::string msg = std::to_string(py_dict.size());
    for (size_t i = 0; i < 4 - msg.size(); i++) {
        msg = "0" + msg;
    }
    msg += py_dict;
    // std::cout << msg << std::endl;
    send_msg(msg);
}

std::string RLClient::read_msg() const {
    char buffer[1024] = {0};
    int valread = read( sock, buffer, 1024);
    std::string msg(buffer);
    // std::cout << "Received: " << msg << std::endl;
    if (msg.find("END") != std::string::npos) {
        std::cout << "Termination due to RL agent" << std::endl;
        exit(0);
    }
    return msg;
}

void RLClient::closeConn() const {
    shutdown(sock, 2);
    close(sock);
    shutdown(sockfd, 2);
    close(sockfd);
}

}
