import React, { useState } from 'react';
import { 
  Container, 
  Paper, 
  Typography, 
  Box, 
  TextField, 
  Button,
  CircularProgress,
  List,
  ListItem,
  ListItemText,
  Divider
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'application/pdf': ['.pdf']
    },
    maxFiles: 1,
    onDrop: acceptedFiles => {
      setFile(acceptedFiles[0]);
    }
  });

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      setLoading(true);
      await axios.post('http://localhost:8000/upload', formData);
      alert('文件上傳成功！');
    } catch (error) {
      console.error('上傳錯誤:', error);
      alert('上傳失敗，請重試');
    } finally {
      setLoading(false);
    }
  };

  const handleQuestion = async () => {
    if (!question) return;

    try {
      setLoading(true);
      const response = await axios.post('http://localhost:8000/query', {
        question: question
      });
      
      const newQA = {
        question: question,
        answer: response.data.answer
      };
      
      setHistory([newQA, ...history]);
      setAnswer(response.data.answer);
      setQuestion('');
    } catch (error) {
      console.error('查詢錯誤:', error);
      alert('查詢失敗，請重試');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          PDF 智能問答系統
        </Typography>

        <Paper
          {...getRootProps()}
          sx={{
            p: 3,
            mb: 3,
            textAlign: 'center',
            backgroundColor: isDragActive ? '#f0f8ff' : 'white',
            cursor: 'pointer'
          }}
        >
          <input {...getInputProps()} />
          <Typography>
            {file 
              ? `已選擇文件: ${file.name}`
              : '拖放 PDF 文件到此處，或點擊選擇文件'
            }
          </Typography>
          {file && (
            <Button
              variant="contained"
              onClick={handleUpload}
              disabled={loading}
              sx={{ mt: 2 }}
            >
              {loading ? <CircularProgress size={24} /> : '上傳文件'}
            </Button>
          )}
        </Paper>

        <Box sx={{ mb: 3 }}>
          <TextField
            fullWidth
            label="請輸入您的問題"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            disabled={loading}
            sx={{ mb: 2 }}
          />
          <Button
            fullWidth
            variant="contained"
            onClick={handleQuestion}
            disabled={loading || !question}
          >
            {loading ? <CircularProgress size={24} /> : '提交問題'}
          </Button>
        </Box>

        {history.length > 0 && (
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              問答歷史
            </Typography>
            <List>
              {history.map((item, index) => (
                <React.Fragment key={index}>
                  <ListItem alignItems="flex-start">
                    <ListItemText
                      primary={`Q: ${item.question}`}
                      secondary={`A: ${item.answer}`}
                    />
                  </ListItem>
                  {index < history.length - 1 && <Divider />}
                </React.Fragment>
              ))}
            </List>
          </Paper>
        )}
      </Box>
    </Container>
  );
}

export default App; 
