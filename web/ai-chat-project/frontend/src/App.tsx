import React, { useState, useEffect } from 'react';
import { Box, CssBaseline, ThemeProvider, createTheme, Typography } from '@mui/material';
import ChatInterface from './components/ChatInterface';
import MessageList from './components/MessageList';
import { Message } from './types/Message';
import './App.css';

const theme = createTheme({
  palette: {
    mode: 'light',
    background: {
      default: '#ffffff',
      paper: '#ffffff',
    },
    primary: {
      main: '#2563eb',
    },
  },
  typography: {
    fontFamily: [
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
  },
});

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchChatHistory = async () => {
    try {
      const response = await fetch('http://localhost:8080/api/chat/history', {
        credentials: 'include'
      });
      if (!response.ok) {
        throw new Error('获取聊天记录失败');
      }
      const data = await response.json();
      setMessages(data);
    } catch (error) {
      console.error('获取聊天记录错误:', error);
    }
  };

  useEffect(() => {
    fetchChatHistory();
  }, []);

  const handleSendMessage = async (content: string) => {
    if (!content.trim()) return;

    const userMessage: Message = {
      id: Date.now(),
      content: content,
      role: 'user',
      createdAt: new Date().toISOString()
    };
    
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8080/api/chat/send', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ content })
      });

      if (!response.ok) {
        throw new Error('发送消息失败');
      }

      const aiResponse = await response.json();
      setMessages(prevMessages => {
        const messagesWithoutTemp = prevMessages.filter(msg => msg.id !== userMessage.id);
        return [...messagesWithoutTemp, {
          ...userMessage,
          id: undefined
        }, aiResponse];
      });
    } catch (error) {
      console.error('发送消息错误:', error);
      setMessages(prevMessages => 
        prevMessages.filter(msg => msg.id !== userMessage.id)
      );
    } finally {
      setLoading(false);
    }
  };

  const handleClearChat = async () => {
    try {
      const response = await fetch('http://localhost:8080/api/chat/clear', {
        method: 'DELETE',
        credentials: 'include'
      });
      
      if (!response.ok) {
        throw new Error('清除聊天记录失败');
      }
      
      setMessages([]);
    } catch (error) {
      console.error('清除聊天记录错误:', error);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ 
        height: '100vh',
        overflow: 'hidden',
        display: 'flex',
        flexDirection: 'column',
        bgcolor: 'background.default'
      }}>
        <Box sx={{
          borderBottom: 1,
          borderColor: 'divider',
          bgcolor: 'background.paper',
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          zIndex: 1000,
        }}>
          <Box sx={{
            maxWidth: '48rem',
            mx: 'auto',
            px: { xs: 2, sm: 3 },
            py: 2,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <Typography 
              variant="h6" 
              component="h1" 
              sx={{ 
                fontWeight: 600,
                color: '#10a37f'
              }}
            >
              Smart Agriculture Q&A Assistant 🌱
            </Typography>
          </Box>
        </Box>

        <Box sx={{ 
          flexGrow: 1, 
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          maxWidth: '48rem',
          width: '100%',
          mx: 'auto',
          position: 'relative',
          mt: '64px', // 为固定标题留出空间
          mb: '56px'  // 更紧凑地为输入框留空间
        }}>
          <Box sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            overflowY: 'auto',
            py: 2,
            px: { xs: 1, sm: 2 }
          }}>
            <MessageList messages={messages} loading={loading} />
          </Box>
        </Box>

        <Box sx={{
          borderTop: 1,
          borderColor: 'divider',
          bgcolor: 'background.paper',
          position: 'fixed',
          bottom: 0,
          left: 0,
          right: 0
        }}>
          <Box sx={{
            maxWidth: '48rem',
            mx: 'auto',
            p: { xs: 0.25, sm: 0.25 }
          }}>
            <ChatInterface 
              onSendMessage={handleSendMessage} 
              onClearChat={handleClearChat}
              loading={loading}
            />
          </Box>
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
