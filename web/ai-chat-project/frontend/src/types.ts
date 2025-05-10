export interface ChatMessage {
  id: number;
  content: string;
  role: 'user' | 'assistant';
  createdAt: string;
} 