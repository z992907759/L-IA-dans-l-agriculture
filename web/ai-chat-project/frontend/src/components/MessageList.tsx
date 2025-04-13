import React, { useState, useRef, useEffect } from 'react';
import { Box, IconButton, Tooltip, CircularProgress, Avatar } from '@mui/material';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import PersonIcon from '@mui/icons-material/Person';

interface Message {
  content: string;
  role: string;
}

interface MessageListProps {
  messages: Message[];
  loading: boolean;
}

const MessageList: React.FC<MessageListProps> = ({ messages, loading }) => {
  const [speakingIndex, setSpeakingIndex] = useState<number | null>(null);
  const [synth, setSynth] = useState<SpeechSynthesis | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (typeof window !== 'undefined') {
      setSynth(window.speechSynthesis);
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const handleSpeak = (text: string, index: number) => {
    if (synth) {
      if (speakingIndex === index) {
        synth.cancel();
        setSpeakingIndex(null);
      } else {
        synth.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.onend = () => setSpeakingIndex(null);
        setSpeakingIndex(index);
        synth.speak(utterance);
      }
    }
  };

  return (
    <Box sx={{ 
      display: 'flex',
      flexDirection: 'column',
      gap: 3,
      p: 2,
      mb: '80px',
      height: 'calc(100vh - 180px)',
      overflow: 'hidden'
    }}>
      <Box sx={{
        display: 'flex',
        flexDirection: 'column',
        gap: 3,
        height: '100%',
        overflowY: 'auto',
        '&::-webkit-scrollbar': {
          width: '4px'
        },
        '&::-webkit-scrollbar-track': {
          background: '#f1f1f1',
          borderRadius: '2px'
        },
        '&::-webkit-scrollbar-thumb': {
          background: '#888',
          borderRadius: '2px',
          '&:hover': {
            background: '#555'
          }
        }
      }}>
        {messages.map((message, index) => (
          <Box
            key={index}
            sx={{
              display: 'flex',
              flexDirection: message.role === 'user' ? 'row-reverse' : 'row',
              alignItems: 'flex-start',
              gap: 2
            }}
          >
            <Avatar
              sx={{
                bgcolor: message.role === 'user' ? '#2563eb' : '#10b981',
                width: 40,
                height: 40,
              }}
            >
              {message.role === 'user' ? <PersonIcon /> : <SmartToyIcon />}
            </Avatar>
            <Box sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: message.role === 'user' ? 'flex-end' : 'flex-start',
              maxWidth: { xs: '70%', sm: '55%' },
            }}>
              <Box sx={{
                backgroundColor: message.role === 'user' ? '#2563eb' : '#f3f4f6',
                color: message.role === 'user' ? '#fff' : '#000',
                borderRadius: '1rem',
                p: 2,
                position: 'relative',
                wordBreak: 'break-all',
                whiteSpace: 'pre-line',
                lineHeight: 1.6,
              }}>
                {message.content}
              </Box>
              {message.role === 'assistant' && (
                <Box sx={{
                  display: 'flex',
                  gap: 1,
                  mt: 1,
                  opacity: 0.7,
                  transition: 'opacity 0.2s',
                  '&:hover': {
                    opacity: 1
                  }
                }}>
                  <Tooltip title="复制">
                    <IconButton
                      size="small"
                      onClick={() => handleCopy(message.content)}
                      sx={{ color: 'text.secondary' }}
                    >
                      <ContentCopyIcon sx={{ fontSize: '1.25rem' }} />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title={speakingIndex === index ? '停止播放' : '播放'}>
                    <IconButton
                      size="small"
                      onClick={() => handleSpeak(message.content, index)}
                      sx={{
                        color: speakingIndex === index ? 'primary.main' : 'text.secondary'
                      }}
                    >
                      <VolumeUpIcon sx={{ fontSize: '1.25rem' }} />
                    </IconButton>
                  </Tooltip>
                </Box>
              )}
            </Box>
          </Box>
        ))}
        {loading && (
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'row',
              alignItems: 'flex-start',
              gap: 2,
            }}
          >
            <Avatar
              sx={{
                bgcolor: '#10b981',
                width: 40,
                height: 40,
              }}
            >
              <SmartToyIcon />
            </Avatar>
            <Box sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 2,
              backgroundColor: '#f3f4f6',
              color: '#000',
              borderRadius: '1rem',
              p: 2,
              maxWidth: { xs: '70%', sm: '55%' },
            }}>
              <CircularProgress size={20} />
              <span>AI is thinking...</span>
            </Box>
          </Box>
        )}
        <div ref={messagesEndRef} />
      </Box>
    </Box>
  );
};

export default MessageList; 