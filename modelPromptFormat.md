Two types of input: 

1. all at once:
```
chat = [
    {"role": "system", "content": "You are a helpful and detailed subtitle reviewer. Rate the subtitle."}, 
    {"role": "user", "content": {"action": "create_review", "subtitle": "Text of movie subtitles"}}, 
    {"role": "assistant", "content": {"rating": 2.5, "review": "The subtitles were somewhat unclear and mistimed."}}
]
```

2. for bigger data per parts:
```
chat = [
    {"role": "system", "content": "You are a helpful and detailed subtitle reviewer. Rate the subtitle."}, 
    
    {"role": "user", "content": {"action": "process_part", "part_number": 1, "subtitle": "Text of movie subtitles"}},
    {"role": "assistant", "content": "Please send the next part."},
    
    {"role": "user", "content": {"action": "process_part", "part_number": 2, "subtitle": "Text of movie subtitles"}},
    {"role": "assistant", "content": "Please send the next part."},

    {"role": "user", "content": {"action": "process_part", "part_number": 3, "subtitle": "Text of movie subtitles", "final_part": true}}, 
    {"role": "assistant", "content": {"rating": 2.5, "review": "The subtitles were somewhat unclear and mistimed."}}
]
```