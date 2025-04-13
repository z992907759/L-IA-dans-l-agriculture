import React, { useState, useRef, KeyboardEvent } from 'react';
import { Box, IconButton, TextField, CircularProgress } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import DeleteIcon from '@mui/icons-material/Delete';

interface ChatInterfaceProps {
  onSendMessage: (message: string) => void;
  onClearChat: () => void;
  loading: boolean;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  onSendMessage,
  onClearChat,
  loading
}) => {
  const [message, setMessage] = useState('');
  const textFieldRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = () => {
    if (message.trim() && !loading) {
      onSendMessage(message);
      setMessage('');
    }
  };

  const handleKeyPress = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSubmit();
    }
  };

  return (
    <Box sx={{ 
      maxWidth: '48rem',
      mx: 'auto',
      width: '100%',
      position: 'relative',
      display: 'flex',
      justifyContent: 'center',
      pb: 2
    }}>
      <Box sx={{
        position: 'relative',
        width: '100%',
        maxWidth: { xs: '85%', sm: '75%', md: '65%' }
      }}>
        <TextField
          fullWidth
          multiline
          minRows={1}
          maxRows={12}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Please enter a question..."
          inputRef={textFieldRef}
          disabled={loading}
          sx={{
            '& .MuiOutlinedInput-root': {
              backgroundColor: '#fff',
              borderRadius: '1rem',
              boxShadow: '0 0 15px rgba(0,0,0,0.1)',
              '& fieldset': {
                borderColor: 'rgba(0,0,0,0.1)',
              },
              '&:hover fieldset': {
                borderColor: 'rgba(0,0,0,0.2)',
              },
              '&.Mui-focused fieldset': {
                borderColor: '#2563eb',
              },
              '& textarea': {
                fontSize: { xs: '0.875rem', sm: '1rem' },
                lineHeight: 1.6,
                padding: { xs: '12px 40px 12px 16px', sm: '14px 80px 14px 20px' },
                minHeight: '24px',
                maxHeight: { xs: '200px', sm: '400px' },
                overflowY: 'auto',
                scrollbarWidth: 'thin',
                scrollbarColor: '#888 #f1f1f1',
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
              }
            }
          }}
        />
        <Box sx={{ 
          position: 'absolute', 
          right: '0.75rem',
          top: '50%',
          transform: 'translateY(-50%)',
          display: 'flex',
          gap: { xs: 0.5, sm: 1 }
        }}>
          <IconButton
            onClick={onClearChat}
            disabled={loading}
            size="small"
            sx={{
              color: 'text.secondary',
              '&:hover': { color: 'error.main' },
              padding: { xs: '4px', sm: '6px' }
            }}
          >
            <DeleteIcon sx={{ fontSize: { xs: '1.125rem', sm: '1.25rem' } }} />
          </IconButton>
          <IconButton
            onClick={handleSubmit}
            disabled={!message.trim() || loading}
            size="small"
            sx={{
              color: message.trim() ? 'primary.main' : 'text.disabled',
              '&:hover': { 
                color: message.trim() ? 'primary.dark' : 'text.disabled' 
              },
              padding: { xs: '4px', sm: '6px' }
            }}
          >
            {loading ? (
              <CircularProgress size={20} />
            ) : (
              <SendIcon sx={{ fontSize: { xs: '1.125rem', sm: '1.25rem' } }} />
            )}
          </IconButton>
        </Box>
      </Box>
    </Box>
  );
};

export default ChatInterface; 