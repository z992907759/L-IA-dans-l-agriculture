package com.aichat.service.impl;

import com.aichat.model.ChatMessage;
import com.aichat.service.ChatService;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.springframework.stereotype.Service;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.HashMap;
import java.util.Map;

@Service
public class ChatServiceImpl implements ChatService {
    private static final Logger logger = LoggerFactory.getLogger(ChatServiceImpl.class);
    private final List<ChatMessage> chatHistory = new ArrayList<>();
    private final String AI_SERVICE_URL = "http://localhost:8000/ask";
    private final ObjectMapper objectMapper = new ObjectMapper();

    @Override
    public ChatMessage sendMessage(String userMessage) {
        logger.info("收到用户消息: {}", userMessage);

        // 保存用户消息
        ChatMessage userChatMessage = new ChatMessage();
        userChatMessage.setContent(userMessage);
        userChatMessage.setRole("user");
        chatHistory.add(userChatMessage);

        try {
            // 调用 AI 服务
            Map<String, String> requestBody = new HashMap<>();
            requestBody.put("prompt", userMessage);
            String requestJson = objectMapper.writeValueAsString(requestBody);
            logger.info("发送请求到 AI 服务: {}", requestJson);

            try (CloseableHttpClient client = HttpClients.createDefault()) {
                HttpPost post = new HttpPost(AI_SERVICE_URL);
                post.setEntity(new StringEntity(requestJson, "UTF-8"));
                post.setHeader("Content-Type", "application/json");
                post.setHeader("Accept", "application/json");

                try (CloseableHttpResponse response = client.execute(post)) {
                    int statusCode = response.getStatusLine().getStatusCode();
                    logger.info("AI 服务响应状态码: {}", statusCode);

                    String jsonResponse = EntityUtils.toString(response.getEntity(), "UTF-8");
                    logger.info("AI 服务响应内容: {}", jsonResponse);

                    if (statusCode != 200) {
                        throw new RuntimeException("AI 服务返回错误状态码: " + statusCode);
                    }

                    Map<String, String> result = objectMapper.readValue(jsonResponse, Map.class);
                    String aiResponse = result.get("answer");  // 注意这里改成了 "answer"
                    if (aiResponse == null || aiResponse.trim().isEmpty()) {
                        throw new RuntimeException("AI 服务返回的响应内容为空");
                    }

                    ChatMessage assistantMessage = new ChatMessage();
                    assistantMessage.setContent(aiResponse);
                    assistantMessage.setRole("assistant");
                    chatHistory.add(assistantMessage);

                    return assistantMessage;
                }
            }
        } catch (Exception e) {
            logger.error("调用 AI 服务出错", e);
            ChatMessage errorMessage = new ChatMessage();
            errorMessage.setContent("抱歉，AI 服务暂时无法响应，请稍后再试。错误信息：" + e.getMessage());
            errorMessage.setRole("assistant");
            chatHistory.add(errorMessage);
            return errorMessage;
        }
    }

    @Override
    public List<ChatMessage> getChatHistory() {
        return chatHistory;
    }

    @Override
    public void clearChatHistory() {
        chatHistory.clear();
    }
}