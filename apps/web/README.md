# Web UI

One page: today's advice, the simulated portfolio, and the track since launch.

```sh
npm install
npm run dev        # http://localhost:5173, proxying /api to http://localhost:8000
npm test           # component tests
npm run typecheck  # tsc --noEmit
npm run build
```

The API must be running for the page to show anything; see the root README.

The wording lives in `src/format.ts` rather than inside the markup, because the wording *is* the
requirement: a decision from an earlier day is reported in the past tense with its date in the
sentence, never as an instruction, and a missing decision is stated as a failure to publish rather
than as a neutral view of the market. Those rules are what the tests assert.
