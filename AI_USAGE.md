# AI usage

## Tools I used, and for which functions

I used ChatGPT to help implement `audit()` in `rawaj/audit.py` and the six decision-metric functions in `rawaj/metrics.py`. I also used it to understand the requirements, review edge cases, and interpret test results. I used VS Code to edit the files and pytest to run the public tests.

## One place where the AI was wrong, incomplete, or not what the spec said

The initial code paste included Markdown code-fence markers, which caused a Python `SyntaxError`. I noticed the problem from the pytest error message, removed the extra markers, and reran the test successfully. This showed me that code must be copied without its surrounding Markdown formatting.

## The function I wrote without AI, and why

I reviewed the public test results and compared the implementation against the task requirements myself. This helped me understand how the expected behavior is checked and why edge cases matter. I used AI assistance for the implementation, so I cannot claim that I wrote a complete function entirely without AI.
