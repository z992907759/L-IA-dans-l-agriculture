package com.aichat.service;

import com.aichat.model.ChatMessage;
import java.util.List;

public interface ChatService {
    ChatMessage sendMessage(String userMessage);
    List<ChatMessage> getChatHistory();
    void clearChatHistory();
} 