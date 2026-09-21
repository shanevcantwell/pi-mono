import {
	type AssistantMessage,
	type AssistantMessageEvent,
	EventStream,
	type Message,
	type Model,
	type ToolResultMessage,
	type UserMessage,
} from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { describe, expect, it } from "vitest";
import { agentLoop, agentLoopContinue, runAgentLoop } from "../src/agent-loop.ts";
import { setDefaultStreamFn } from "../src/index.ts";
import type { AgentContext, AgentEvent, AgentLoopConfig, AgentMessage, AgentTool } from "../src/types.ts";

// Mock stream for testing - mimics MockAssistantStream
class MockAssistantStream extends EventStream<AssistantMessageEvent, AssistantMessage> {
	constructor() {
		super(
			(event) => event.type === "done" || event.type === "error",
			(event) => {
				if (event.type === "done") return event.message;
				if (event.type === "error") return event.error;
				throw new Error("Unexpected event type");
			},
		);
	}
}

function createUsage() {
	return {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function createModel(): Model<"openai-responses"> {
	return {
		id: "mock",
		name: "mock",
		api: "openai-responses",
		provider: "openai",
		baseUrl: "https://example.invalid",
		reasoning: false,
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 8192,
		maxTokens: 2048,
	};
}

function createAssistantMessage(
	content: AssistantMessage["content"],
	stopReason: AssistantMessage["stopReason"] = "stop",
): AssistantMessage {
	return {
		role: "assistant",
		content,
		api: "openai-responses",
		provider: "openai",
		model: "mock",
		usage: createUsage(),
		stopReason,
		timestamp: Date.now(),
	};
}

function createUserMessage(text: string): UserMessage {
	return {
		role: "user",
		content: text,
		timestamp: Date.now(),
	};
}

// Simple identity converter for tests - just passes through standard messages
function identityConverter(messages: AgentMessage[]): Message[] {
	return messages.filter(
		(m) => m.role === "system" || m.role === "user" || m.role === "assistant" || m.role === "toolResult",
	) as Message[];
}

function expectCompleteToolTranscript(messages: readonly Message[]): void {
	const toolCallIds = messages.flatMap((message) =>
		message.role === "assistant"
			? message.content.flatMap((part) => (part.type === "toolCall" ? [part.id] : []))
			: [],
	);
	const toolResultIds = messages.flatMap((message) => (message.role === "toolResult" ? [message.toolCallId] : []));
	expect(toolResultIds).toEqual(toolCallIds);
	expect(new Set(toolResultIds).size).toBe(toolResultIds.length);
}

describe("default stream function compatibility", () => {
	it("uses the configured default when a legacy caller omits streamFn", async () => {
		let calls = 0;
		setDefaultStreamFn(() => {
			calls++;
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				stream.push({
					type: "done",
					reason: "stop",
					message: createAssistantMessage([{ type: "text", text: "fallback" }]),
				});
			});
			return stream;
		});

		try {
			const context: AgentContext = { messages: [], tools: [] };
			const config: AgentLoopConfig = { model: createModel(), convertToLlm: identityConverter };
			const stream = Reflect.apply(agentLoop, undefined, [
				[createUserMessage("Hello")],
				context,
				config,
				undefined,
			]) as ReturnType<typeof agentLoop>;

			await stream.result();
			expect(calls).toBe(1);
		} finally {
			setDefaultStreamFn(undefined);
		}
	});
});

describe("agentLoop with AgentMessage", () => {
	it("should emit events with AgentMessage types", async () => {
		const context: AgentContext = {
			messages: [],
			tools: [],
		};

		const userPrompt: AgentMessage = createUserMessage("Hello");

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
		};

		const streamFn = () => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				const message = createAssistantMessage([{ type: "text", text: "Hi there!" }]);
				stream.push({ type: "done", reason: "stop", message });
			});
			return stream;
		};

		const events: AgentEvent[] = [];
		const stream = agentLoop([userPrompt], context, config, undefined, streamFn);

		for await (const event of stream) {
			events.push(event);
		}

		const messages = await stream.result();

		// Should have user message and assistant message
		expect(messages.length).toBe(2);
		expect(messages[0].role).toBe("user");
		expect(messages[1].role).toBe("assistant");

		// Verify event sequence
		const eventTypes = events.map((e) => e.type);
		expect(eventTypes).toContain("agent_start");
		expect(eventTypes).toContain("turn_start");
		expect(eventTypes).toContain("message_start");
		expect(eventTypes).toContain("message_end");
		expect(eventTypes).toContain("turn_end");
		expect(eventTypes).toContain("agent_end");
	});

	it("should build provider context exclusively from transcript messages", async () => {
		const initialSystem: AgentMessage = {
			role: "system",
			content: "Transcript prompt",
			toolsAdded: [],
			timestamp: 1,
		};
		const context: AgentContext = {
			messages: [],
			tools: [],
		};
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
		};
		const stream = agentLoop(
			[initialSystem, createUserMessage("Hello")],
			context,
			config,
			undefined,
			(_model, providerContext) => {
				// The provider receives a transcript: no top-level prompt or tool fields.
				expect(Object.keys(providerContext)).toEqual(["messages"]);
				expect(providerContext.messages[0]).toBe(initialSystem);
				const response = new MockAssistantStream();
				queueMicrotask(() => {
					response.push({
						type: "done",
						reason: "stop",
						message: createAssistantMessage([{ type: "text", text: "done" }]),
					});
				});
				return response;
			},
		);

		await stream.result();
	});

	it("should handle custom message types via convertToLlm", async () => {
		// Create a custom message type
		interface CustomNotification {
			role: "notification";
			text: string;
			timestamp: number;
		}

		const notification: CustomNotification = {
			role: "notification",
			text: "This is a notification",
			timestamp: Date.now(),
		};

		const context: AgentContext = {
			messages: [notification as unknown as AgentMessage], // Custom message in context
			tools: [],
		};

		const userPrompt: AgentMessage = createUserMessage("Hello");

		let convertedMessages: Message[] = [];
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: (messages) => {
				// Filter out notifications, convert rest
				convertedMessages = messages
					.filter((m) => (m as { role: string }).role !== "notification")
					.filter((m) => m.role === "user" || m.role === "assistant" || m.role === "toolResult") as Message[];
				return convertedMessages;
			},
		};

		const streamFn = () => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				const message = createAssistantMessage([{ type: "text", text: "Response" }]);
				stream.push({ type: "done", reason: "stop", message });
			});
			return stream;
		};

		const events: AgentEvent[] = [];
		const stream = agentLoop([userPrompt], context, config, undefined, streamFn);

		for await (const event of stream) {
			events.push(event);
		}

		// The notification should have been filtered out in convertToLlm
		expect(convertedMessages.length).toBe(1); // Only user message
		expect(convertedMessages[0].role).toBe("user");
	});

	it("should apply transformContext before convertToLlm", async () => {
		const context: AgentContext = {
			messages: [
				createUserMessage("old message 1"),
				createAssistantMessage([{ type: "text", text: "old response 1" }]),
				createUserMessage("old message 2"),
				createAssistantMessage([{ type: "text", text: "old response 2" }]),
			],
			tools: [],
		};

		const userPrompt: AgentMessage = createUserMessage("new message");

		let transformedMessages: AgentMessage[] = [];
		let convertedMessages: Message[] = [];

		const config: AgentLoopConfig = {
			model: createModel(),
			transformContext: async (messages) => {
				// Keep only last 2 messages (prune old ones)
				transformedMessages = messages.slice(-2);
				return transformedMessages;
			},
			convertToLlm: (messages) => {
				convertedMessages = messages.filter(
					(m) => m.role === "user" || m.role === "assistant" || m.role === "toolResult",
				) as Message[];
				return convertedMessages;
			},
		};

		const streamFn = () => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				const message = createAssistantMessage([{ type: "text", text: "Response" }]);
				stream.push({ type: "done", reason: "stop", message });
			});
			return stream;
		};

		const stream = agentLoop([userPrompt], context, config, undefined, streamFn);

		for await (const _ of stream) {
			// consume
		}

		// transformContext should have been called first, keeping only last 2
		expect(transformedMessages.length).toBe(2);
		// Then convertToLlm receives the pruned messages
		expect(convertedMessages.length).toBe(2);
	});

	it("should handle tool calls and results", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const executed: string[] = [];
		const toolUsage = {
			input: 1,
			output: 2,
			cacheRead: 3,
			cacheWrite: 4,
			totalTokens: 10,
			cost: { input: 0.1, output: 0.2, cacheRead: 0.3, cacheWrite: 0.4, total: 1 },
		};
		const patchedToolUsage = {
			input: 5,
			output: 6,
			cacheRead: 7,
			cacheWrite: 8,
			totalTokens: 26,
			cost: { input: 0.5, output: 0.6, cacheRead: 0.7, cacheWrite: 0.8, total: 2.6 },
		};
		let observedToolUsage: typeof toolUsage | undefined;
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value);
				return {
					content: [{ type: "text", text: `echoed: ${params.value}` }],
					details: { value: params.value },
					usage: toolUsage,
				};
			},
		};

		const context: AgentContext = {
			messages: [],
			tools: [tool],
		};

		const userPrompt: AgentMessage = createUserMessage("echo something");

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			afterToolCall: async ({ result }) => {
				observedToolUsage = result.usage;
				return { usage: patchedToolUsage };
			},
		};

		let callIndex = 0;
		const streamFn = () => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				if (callIndex === 0) {
					// First call: return tool call
					const message = createAssistantMessage(
						[{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "hello" } }],
						"toolUse",
					);
					stream.push({ type: "done", reason: "toolUse", message });
				} else {
					// Second call: return final response
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					stream.push({ type: "done", reason: "stop", message });
				}
				callIndex++;
			});
			return stream;
		};

		const events: AgentEvent[] = [];
		const stream = agentLoop([userPrompt], context, config, undefined, streamFn);

		for await (const event of stream) {
			events.push(event);
		}

		// Tool should have been executed
		expect(executed).toEqual(["hello"]);

		// Should have tool execution events
		const toolStart = events.find((e) => e.type === "tool_execution_start");
		const toolEnd = events.find((e) => e.type === "tool_execution_end");
		expect(toolStart).toBeDefined();
		expect(toolEnd).toBeDefined();
		if (toolEnd?.type === "tool_execution_end") {
			expect(toolEnd.isError).toBe(false);
		}
		expect(observedToolUsage).toEqual(toolUsage);
		const messages = await stream.result();
		const toolResult = messages.find((message) => message.role === "toolResult");
		expect(toolResult?.role === "toolResult" ? toolResult.usage : undefined).toEqual(patchedToolUsage);
	});

	it("should not execute tool calls from a length-truncated assistant message", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const executed: string[] = [];
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value);
				return {
					content: [{ type: "text", text: `echoed: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			messages: [],
			tools: [tool],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
		};

		let callIndex = 0;
		const streamFn = () => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				if (callIndex === 0) {
					// Output hit the token limit mid tool call. The salvage parser can
					// produce arguments that validate but are silently truncated, so
					// nothing in this message may execute.
					const message = createAssistantMessage(
						[{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "hel" } }],
						"length",
					);
					stream.push({ type: "done", reason: "length", message });
				} else {
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					stream.push({ type: "done", reason: "stop", message });
				}
				callIndex++;
			});
			return stream;
		};

		const events: AgentEvent[] = [];
		const stream = agentLoop([createUserMessage("echo something")], context, config, undefined, streamFn);
		for await (const event of stream) {
			events.push(event);
		}

		// The tool must never execute with potentially truncated arguments.
		expect(executed).toEqual([]);

		const toolEnd = events.find((e) => e.type === "tool_execution_end");
		expect(toolEnd).toBeDefined();
		if (toolEnd?.type === "tool_execution_end") {
			expect(toolEnd.isError).toBe(true);
			const text = toolEnd.result.content.find((c: { type: string }) => c.type === "text");
			expect(text && "text" in text ? text.text : "").toContain("output token limit");
		}

		// The loop continues so the model can re-issue the tool call.
		expect(callIndex).toBe(2);
		const messages = await stream.result();
		expect(messages[messages.length - 1].role).toBe("assistant");
	});

	it("should execute mutated beforeToolCall args without revalidation", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const executed: Array<string | number> = [];
		const tool: AgentTool<typeof toolSchema, { value: string | number }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value as string | number);
				return {
					content: [{ type: "text", text: `echoed: ${String(params.value)}` }],
					details: { value: params.value as string | number },
				};
			},
		};

		const context: AgentContext = {
			messages: [],
			tools: [tool],
		};

		const userPrompt: AgentMessage = createUserMessage("echo something");

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			beforeToolCall: async ({ args }) => {
				const mutableArgs = args as { value: string | number };
				mutableArgs.value = 123;
				return undefined;
			},
		};

		let callIndex = 0;
		const streamFn = () => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				if (callIndex === 0) {
					const message = createAssistantMessage(
						[{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "hello" } }],
						"toolUse",
					);
					stream.push({ type: "done", reason: "toolUse", message });
				} else {
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					stream.push({ type: "done", reason: "stop", message });
				}
				callIndex++;
			});
			return stream;
		};

		const stream = agentLoop([userPrompt], context, config, undefined, streamFn);
		for await (const _event of stream) {
			// consume
		}

		expect(executed).toEqual([123]);
	});

	it("should prepare tool arguments for validation", async () => {
		const replaceSchema = Type.Object({ oldText: Type.String(), newText: Type.String() });
		const toolSchema = Type.Object({ edits: Type.Array(replaceSchema) });
		const executed: Array<Array<{ oldText: string; newText: string }>> = [];
		const tool: AgentTool<typeof toolSchema, { count: number }> = {
			name: "edit",
			label: "Edit",
			description: "Edit tool",
			parameters: toolSchema,
			prepareArguments(args) {
				if (!args || typeof args !== "object") {
					return args as { edits: { oldText: string; newText: string }[] };
				}
				const input = args as {
					edits?: Array<{ oldText: string; newText: string }>;
					oldText?: string;
					newText?: string;
				};
				if (typeof input.oldText !== "string" || typeof input.newText !== "string") {
					return args as { edits: { oldText: string; newText: string }[] };
				}
				return {
					edits: [...(input.edits ?? []), { oldText: input.oldText, newText: input.newText }],
				};
			},
			async execute(_toolCallId, params) {
				executed.push(params.edits);
				return {
					content: [{ type: "text", text: `edited ${params.edits.length}` }],
					details: { count: params.edits.length },
				};
			},
		};

		const context: AgentContext = {
			messages: [],
			tools: [tool],
		};

		const userPrompt: AgentMessage = createUserMessage("edit something");
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
		};

		let callIndex = 0;
		const streamFn = () => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				if (callIndex === 0) {
					const message = createAssistantMessage(
						[
							{
								type: "toolCall",
								id: "tool-1",
								name: "edit",
								arguments: { oldText: "before", newText: "after" },
							},
						],
						"toolUse",
					);
					stream.push({ type: "done", reason: "toolUse", message });
				} else {
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					stream.push({ type: "done", reason: "stop", message });
				}
				callIndex++;
			});
			return stream;
		};

		const stream = agentLoop([userPrompt], context, config, undefined, streamFn);
		for await (const _event of stream) {
			// consume
		}

		expect(executed).toEqual([[{ oldText: "before", newText: "after" }]]);
	});

	it("should emit tool_execution_end in completion order but persist tool results in source order", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		let firstResolved = false;
		let parallelObserved = false;
		let releaseFirst: (() => void) | undefined;
		const firstDone = new Promise<void>((resolve) => {
			releaseFirst = resolve;
		});

		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				if (params.value === "first") {
					await firstDone;
					firstResolved = true;
				}
				if (params.value === "second" && !firstResolved) {
					parallelObserved = true;
				}
				return {
					content: [{ type: "text", text: `echoed: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			messages: [],
			tools: [tool],
		};

		const userPrompt: AgentMessage = createUserMessage("echo both");
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			toolExecution: "parallel",
		};

		let callIndex = 0;
		const stream = agentLoop([userPrompt], context, config, undefined, () => {
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				if (callIndex === 0) {
					const message = createAssistantMessage(
						[
							{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "first" } },
							{ type: "toolCall", id: "tool-2", name: "echo", arguments: { value: "second" } },
						],
						"toolUse",
					);
					mockStream.push({ type: "done", reason: "toolUse", message });
					setTimeout(() => releaseFirst?.(), 20);
				} else {
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					mockStream.push({ type: "done", reason: "stop", message });
				}
				callIndex++;
			});
			return mockStream;
		});

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		const toolExecutionEndIds = events.flatMap((event) => {
			if (event.type !== "tool_execution_end") {
				return [];
			}
			return [event.toolCallId];
		});
		const toolResultIds = events.flatMap((event) => {
			if (event.type !== "message_end" || event.message.role !== "toolResult") {
				return [];
			}
			return [event.message.toolCallId];
		});
		const turnToolResultIds = events.flatMap((event) => {
			if (event.type !== "turn_end") {
				return [];
			}
			return event.toolResults.map((toolResult) => toolResult.toolCallId);
		});

		expect(parallelObserved).toBe(true);
		expect(toolExecutionEndIds).toEqual(["tool-2", "tool-1"]);
		expect(toolResultIds).toEqual(["tool-1", "tool-2"]);
		expect(turnToolResultIds).toEqual(["tool-1", "tool-2"]);
	});

	it("should skip a sequential batch when steering was queued during assistant streaming", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const preflights: string[] = [];
		const executions: string[] = [];
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executions.push(params.value);
				return { content: [{ type: "text", text: params.value }], details: { value: params.value } };
			},
		};
		let steeringReady = false;
		let steeringDelivered = false;
		let llmCalls = 0;
		const events: AgentEvent[] = [];
		const stream = agentLoop(
			[createUserMessage("start")],
			{ messages: [], tools: [tool] },
			{
				model: createModel(),
				convertToLlm: identityConverter,
				toolExecution: "sequential",
				beforeToolCall: async ({ toolCall }) => {
					preflights.push(toolCall.id);
					return undefined;
				},
				getSteeringMessages: async () => {
					if (!steeringReady || steeringDelivered) return [];
					steeringDelivered = true;
					return [createUserMessage("replan")];
				},
			},
			undefined,
			(_model, context) => {
				const mockStream = new MockAssistantStream();
				queueMicrotask(() => {
					if (llmCalls++ === 0) {
						steeringReady = true;
						mockStream.push({
							type: "done",
							reason: "toolUse",
							message: createAssistantMessage(
								[
									{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "first" } },
									{ type: "toolCall", id: "tool-2", name: "echo", arguments: { value: "second" } },
								],
								"toolUse",
							),
						});
					} else {
						expect(context.messages.map((message) => message.role).slice(-3)).toEqual([
							"toolResult",
							"toolResult",
							"user",
						]);
						mockStream.push({
							type: "done",
							reason: "stop",
							message: createAssistantMessage([{ type: "text", text: "replanned" }]),
						});
					}
				});
				return mockStream;
			},
		);

		for await (const event of stream) events.push(event);
		const messages = await stream.result();

		expect(preflights).toEqual([]);
		expect(executions).toEqual([]);
		expect(events.some((event) => event.type === "tool_execution_start")).toBe(false);
		expect(messages.filter((message) => message.role === "toolResult").map((message) => message.toolCallId)).toEqual([
			"tool-1",
			"tool-2",
		]);
	});

	it("should skip a single parallel call when steering was queued during assistant streaming", async () => {
		const toolSchema = Type.Object({});
		let preflights = 0;
		let executions = 0;
		const tool: AgentTool<typeof toolSchema, Record<string, never>> = {
			name: "work",
			label: "Work",
			description: "Work tool",
			parameters: toolSchema,
			async execute() {
				executions++;
				return { content: [{ type: "text", text: "done" }], details: {} };
			},
		};
		let steeringReady = false;
		let steeringDelivered = false;
		let llmCalls = 0;
		const events: AgentEvent[] = [];
		const stream = agentLoop(
			[createUserMessage("start")],
			{ messages: [], tools: [tool] },
			{
				model: createModel(),
				convertToLlm: identityConverter,
				toolExecution: "parallel",
				beforeToolCall: async () => {
					preflights++;
					return undefined;
				},
				getSteeringMessages: async () => {
					if (!steeringReady || steeringDelivered) return [];
					steeringDelivered = true;
					return [createUserMessage("replan")];
				},
			},
			undefined,
			() => {
				const mockStream = new MockAssistantStream();
				queueMicrotask(() => {
					if (llmCalls++ === 0) {
						steeringReady = true;
						mockStream.push({
							type: "done",
							reason: "toolUse",
							message: createAssistantMessage(
								[{ type: "toolCall", id: "tool-1", name: "work", arguments: {} }],
								"toolUse",
							),
						});
					} else {
						mockStream.push({
							type: "done",
							reason: "stop",
							message: createAssistantMessage([{ type: "text", text: "replanned" }]),
						});
					}
				});
				return mockStream;
			},
		);

		for await (const event of stream) events.push(event);
		const messages = await stream.result();

		expect(preflights).toBe(0);
		expect(executions).toBe(0);
		expect(events.some((event) => event.type === "tool_execution_start")).toBe(false);
		expect(messages.filter((message) => message.role === "toolResult").map((message) => message.toolCallId)).toEqual([
			"tool-1",
		]);
	});

	it("should skip pending parallel calls that have not launched when steering arrives during preflight", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const preflights: string[] = [];
		const executed: string[] = [];
		let releasePreflight: (() => void) | undefined;
		let markPreflightStarted: (() => void) | undefined;
		const preflightStarted = new Promise<void>((resolve) => {
			markPreflightStarted = resolve;
		});
		const preflightGate = new Promise<void>((resolve) => {
			releasePreflight = resolve;
		});
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value);
				return { content: [{ type: "text", text: params.value }], details: { value: params.value } };
			},
		};
		const context: AgentContext = { messages: [], tools: [tool] };
		let steeringReady = false;
		let steeringDelivered = false;
		let llmCalls = 0;
		const stream = agentLoop(
			[createUserMessage("start")],
			context,
			{
				model: createModel(),
				convertToLlm: identityConverter,
				toolExecution: "parallel",
				beforeToolCall: async ({ toolCall }) => {
					preflights.push(toolCall.id);
					if (toolCall.id === "tool-1") {
						markPreflightStarted?.();
						await preflightGate;
					}
					return undefined;
				},
				getSteeringMessages: async () => {
					if (!steeringReady || steeringDelivered) return [];
					steeringDelivered = true;
					return [createUserMessage("replan")];
				},
			},
			undefined,
			(_model, ctx) => {
				const mockStream = new MockAssistantStream();
				queueMicrotask(() => {
					if (llmCalls++ === 0) {
						mockStream.push({
							type: "done",
							reason: "toolUse",
							message: createAssistantMessage(
								[
									{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "first" } },
									{ type: "toolCall", id: "tool-2", name: "echo", arguments: { value: "second" } },
								],
								"toolUse",
							),
						});
					} else {
						expect(ctx.messages.map((message) => message.role).slice(-3)).toEqual([
							"toolResult",
							"toolResult",
							"user",
						]);
						mockStream.push({
							type: "done",
							reason: "stop",
							message: createAssistantMessage([{ type: "text", text: "replanned" }]),
						});
					}
				});
				return mockStream;
			},
		);

		await preflightStarted;
		steeringReady = true;
		releasePreflight?.();
		const messages = await stream.result();

		expect(preflights).toEqual(["tool-1"]);
		expect(executed).toEqual([]);
		expect(
			messages
				.filter((message) => message.role === "toolResult")
				.map((message) => [message.toolCallId, message.isError]),
		).toEqual([
			["tool-1", true],
			["tool-2", true],
		]);
	});

	it("should skip the current and later sequential calls when steering arrives during preflight", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const preflights: string[] = [];
		const executed: string[] = [];
		let releasePreflight: (() => void) | undefined;
		let markPreflightStarted: (() => void) | undefined;
		const preflightStarted = new Promise<void>((resolve) => {
			markPreflightStarted = resolve;
		});
		const preflightGate = new Promise<void>((resolve) => {
			releasePreflight = resolve;
		});
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value);
				return { content: [{ type: "text", text: params.value }], details: { value: params.value } };
			},
		};
		let steeringReady = false;
		let steeringDelivered = false;
		let llmCalls = 0;
		let replanTranscriptWasValid = false;
		const events: AgentEvent[] = [];
		const stream = agentLoop(
			[createUserMessage("start")],
			{ messages: [], tools: [tool] },
			{
				model: createModel(),
				convertToLlm: identityConverter,
				toolExecution: "sequential",
				beforeToolCall: async ({ toolCall }) => {
					preflights.push(toolCall.id);
					markPreflightStarted?.();
					await preflightGate;
					return undefined;
				},
				getSteeringMessages: async () => {
					if (!steeringReady || steeringDelivered) return [];
					steeringDelivered = true;
					return [createUserMessage("replan")];
				},
			},
			undefined,
			(_model, providerContext) => {
				const mockStream = new MockAssistantStream();
				queueMicrotask(() => {
					if (llmCalls++ === 0) {
						mockStream.push({
							type: "done",
							reason: "toolUse",
							message: createAssistantMessage(
								[
									{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "first" } },
									{ type: "toolCall", id: "tool-2", name: "echo", arguments: { value: "second" } },
								],
								"toolUse",
							),
						});
					} else {
						expectCompleteToolTranscript(providerContext.messages);
						expect(providerContext.messages.map((message) => message.role).slice(-3)).toEqual([
							"toolResult",
							"toolResult",
							"user",
						]);
						replanTranscriptWasValid = true;
						mockStream.push({
							type: "done",
							reason: "stop",
							message: createAssistantMessage([{ type: "text", text: "replanned" }]),
						});
					}
				});
				return mockStream;
			},
		);

		await preflightStarted;
		steeringReady = true;
		releasePreflight?.();
		for await (const event of stream) events.push(event);
		const messages = await stream.result();
		const results = messages.filter((message): message is ToolResultMessage => message.role === "toolResult");

		expect(preflights).toEqual(["tool-1"]);
		expect(executed).toEqual([]);
		expect(events.filter((event) => event.type === "tool_execution_start")).toEqual([]);
		expect(results.map((result) => [result.toolCallId, result.isError])).toEqual([
			["tool-1", true],
			["tool-2", true],
		]);
		expect(replanTranscriptWasValid).toBe(true);
		expectCompleteToolTranscript(identityConverter(messages));
	});

	it("should not treat already-launched parallel calls as pending when steering arrives", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const executed: string[] = [];
		let steeringReady = false;
		let steeringDelivered = false;
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value);
				if (params.value === "first") {
					queueMicrotask(() => {
						steeringReady = true;
					});
				}
				if (params.value === "second") await new Promise((resolve) => setTimeout(resolve, 10));
				return { content: [{ type: "text", text: params.value }], details: { value: params.value } };
			},
		};
		const context: AgentContext = { messages: [], tools: [tool] };
		let llmCalls = 0;
		const stream = agentLoop(
			[createUserMessage("start")],
			context,
			{
				model: createModel(),
				convertToLlm: identityConverter,
				toolExecution: "parallel",
				getSteeringMessages: async () => {
					if (!steeringReady || steeringDelivered) return [];
					steeringDelivered = true;
					return [createUserMessage("replan")];
				},
			},
			undefined,
			() => {
				const mockStream = new MockAssistantStream();
				queueMicrotask(() => {
					mockStream.push(
						llmCalls++ === 0
							? {
									type: "done",
									reason: "toolUse",
									message: createAssistantMessage(
										[
											{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "first" } },
											{ type: "toolCall", id: "tool-2", name: "echo", arguments: { value: "second" } },
										],
										"toolUse",
									),
								}
							: {
									type: "done",
									reason: "stop",
									message: createAssistantMessage([{ type: "text", text: "replanned" }]),
								},
					);
				});
				return mockStream;
			},
		);

		const messages = await stream.result();

		expect(executed).toEqual(["first", "second"]);
		expect(messages.filter((message) => message.role === "toolResult").map((message) => message.isError)).toEqual([
			false,
			false,
		]);
		expect(steeringDelivered).toBe(true);
	});

	it("should skip unstarted tool calls and replan when steering arrives during a sequential batch", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const executed: string[] = [];
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value);
				return {
					content: [{ type: "text", text: `ok:${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			messages: [],
			tools: [tool],
		};

		const userPrompt: AgentMessage = createUserMessage("start");
		const queuedUserMessage: AgentMessage = createUserMessage("interrupt");

		let queuedDelivered = false;
		let callIndex = 0;
		let sawInterruptInContext = false;

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			toolExecution: "sequential",
			getSteeringMessages: async () => {
				// Return steering message after tool execution has started.
				if (executed.length >= 1 && !queuedDelivered) {
					queuedDelivered = true;
					return [queuedUserMessage];
				}
				return [];
			},
		};

		const events: AgentEvent[] = [];
		const stream = agentLoop([userPrompt], context, config, undefined, (_model, ctx, _options) => {
			// Check if interrupt message is in context on second call
			if (callIndex === 1) {
				sawInterruptInContext = ctx.messages.some(
					(m) => m.role === "user" && typeof m.content === "string" && m.content === "interrupt",
				);
			}

			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				if (callIndex === 0) {
					// First call: return two tool calls
					const message = createAssistantMessage(
						[
							{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "first" } },
							{ type: "toolCall", id: "tool-2", name: "echo", arguments: { value: "second" } },
						],
						"toolUse",
					);
					mockStream.push({ type: "done", reason: "toolUse", message });
				} else {
					// Second call: return final response
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					mockStream.push({ type: "done", reason: "stop", message });
				}
				callIndex++;
			});
			return mockStream;
		});

		for await (const event of stream) {
			events.push(event);
		}
		const resultingMessages = await stream.result();

		// The active call reaches its safe boundary; the pending call is never started.
		expect(executed).toEqual(["first"]);

		const toolEnds = events.filter(
			(e): e is Extract<AgentEvent, { type: "tool_execution_end" }> => e.type === "tool_execution_end",
		);
		expect(toolEnds).toHaveLength(1);
		expect(toolEnds[0].isError).toBe(false);
		const skippedResult = resultingMessages.find(
			(message): message is ToolResultMessage => message.role === "toolResult" && message.toolCallId === "tool-2",
		);
		expect(skippedResult?.isError).toBe(true);
		expect(skippedResult?.content).toEqual([
			{ type: "text", text: "Skipped because a steering message invalidated this tool batch." },
		]);

		// Every assistant tool call has a source-ordered result before steering is delivered,
		// keeping the next provider request protocol-valid while asking it to replan.
		const eventSequence = events.flatMap((event) => {
			if (event.type !== "message_start") return [];
			if (event.message.role === "toolResult") return [`tool:${event.message.toolCallId}`];
			if (event.message.role === "user" && typeof event.message.content === "string") {
				return [event.message.content];
			}
			return [];
		});
		expect(eventSequence).toEqual(expect.arrayContaining(["tool:tool-1", "tool:tool-2", "interrupt"]));
		expect(eventSequence.indexOf("tool:tool-1")).toBeLessThan(eventSequence.indexOf("tool:tool-2"));
		expect(eventSequence.indexOf("tool:tool-2")).toBeLessThan(eventSequence.indexOf("interrupt"));

		const replanningContext = resultingMessages;
		const assistantToolCallIds = replanningContext.flatMap((message) =>
			message.role === "assistant"
				? message.content.flatMap((part) => (part.type === "toolCall" ? [part.id] : []))
				: [],
		);
		const toolResultIds = replanningContext.flatMap((message) =>
			message.role === "toolResult" ? [message.toolCallId] : [],
		);
		expect(toolResultIds).toEqual(assistantToolCallIds);
		expect(sawInterruptInContext).toBe(true);
	});

	it("should deliver steering queued from turn_end after a terminating tool", async () => {
		const toolSchema = Type.Object({});
		const tool: AgentTool<typeof toolSchema, Record<string, never>> = {
			name: "terminate",
			label: "Terminate",
			description: "Terminate the current batch",
			parameters: toolSchema,
			async execute() {
				return { content: [{ type: "text", text: "terminated" }], details: {}, terminate: true };
			},
		};
		const context: AgentContext = { messages: [], tools: [tool] };
		let turnEnded = false;
		let steeringDelivered = false;
		let llmCalls = 0;
		let sawCorrection = false;
		const messages = await runAgentLoop(
			[createUserMessage("start")],
			context,
			{
				model: createModel(),
				convertToLlm: identityConverter,
				toolExecution: "sequential",
				getSteeringMessages: async () => {
					if (!turnEnded || steeringDelivered) return [];
					steeringDelivered = true;
					return [createUserMessage("replan after termination")];
				},
			},
			async (event) => {
				if (event.type === "turn_end") turnEnded = true;
			},
			undefined,
			(_model, providerContext) => {
				const mockStream = new MockAssistantStream();
				queueMicrotask(() => {
					if (llmCalls++ === 0) {
						mockStream.push({
							type: "done",
							reason: "toolUse",
							message: createAssistantMessage(
								[{ type: "toolCall", id: "tool-1", name: "terminate", arguments: {} }],
								"toolUse",
							),
						});
					} else {
						sawCorrection = providerContext.messages.some(
							(message) => message.role === "user" && message.content === "replan after termination",
						);
						mockStream.push({
							type: "done",
							reason: "stop",
							message: createAssistantMessage([{ type: "text", text: "replanned" }]),
						});
					}
				});
				return mockStream;
			},
		);

		expect(llmCalls).toBe(2);
		expect(steeringDelivered).toBe(true);
		expect(sawCorrection).toBe(true);
		expect(
			messages.some((message) => message.role === "user" && message.content === "replan after termination"),
		).toBe(true);
	});

	it("should replan a no-tool turn when steering arrives during a true stop callback", async () => {
		let releaseStopCheck: (() => void) | undefined;
		let markStopCheckStarted: (() => void) | undefined;
		const stopCheckStarted = new Promise<void>((resolve) => {
			markStopCheckStarted = resolve;
		});
		const stopCheckGate = new Promise<void>((resolve) => {
			releaseStopCheck = resolve;
		});
		let steeringReady = false;
		let steeringDelivered = false;
		let stopChecks = 0;
		let llmCalls = 0;
		let sawSteeringOnReplan = false;
		const loopPromise = runAgentLoop(
			[createUserMessage("start")],
			{ messages: [], tools: [] },
			{
				model: createModel(),
				convertToLlm: identityConverter,
				getSteeringMessages: async () => {
					if (!steeringReady || steeringDelivered) return [];
					steeringDelivered = true;
					return [createUserMessage("late steering")];
				},
				shouldStopAfterTurn: async () => {
					stopChecks++;
					if (stopChecks === 1) {
						markStopCheckStarted?.();
						await stopCheckGate;
					}
					return true;
				},
			},
			async () => {},
			undefined,
			(_model, providerContext) => {
				const mockStream = new MockAssistantStream();
				queueMicrotask(() => {
					llmCalls++;
					if (llmCalls === 2) {
						sawSteeringOnReplan = providerContext.messages.some(
							(message) => message.role === "user" && message.content === "late steering",
						);
					}
					mockStream.push({
						type: "done",
						reason: "stop",
						message: createAssistantMessage([{ type: "text", text: `response ${llmCalls}` }]),
					});
				});
				return mockStream;
			},
		);

		await stopCheckStarted;
		steeringReady = true;
		releaseStopCheck?.();
		const messages = await loopPromise;

		expect(llmCalls).toBe(2);
		expect(stopChecks).toBe(2);
		expect(sawSteeringOnReplan).toBe(true);
		expect(messages.filter((message) => message.role === "user" && message.content === "late steering")).toHaveLength(
			1,
		);
	});

	it("should prioritize steering queued during follow-up polling and defer the drained follow-up", async () => {
		const toolSchema = Type.Object({});
		let executions = 0;
		const tool: AgentTool<typeof toolSchema, Record<string, never>> = {
			name: "terminate",
			label: "Terminate",
			description: "Terminate the current batch",
			parameters: toolSchema,
			async execute() {
				executions++;
				return { content: [{ type: "text", text: "terminated" }], details: {}, terminate: true };
			},
		};
		let releaseFollowUp: (() => void) | undefined;
		let markFollowUpStarted: (() => void) | undefined;
		const followUpStarted = new Promise<void>((resolve) => {
			markFollowUpStarted = resolve;
		});
		const followUpGate = new Promise<void>((resolve) => {
			releaseFollowUp = resolve;
		});
		let steeringReady = false;
		let steeringDelivered = false;
		let followUpPolls = 0;
		let llmCalls = 0;
		const userMessagesByCall: string[][] = [];
		const loopPromise = runAgentLoop(
			[createUserMessage("start")],
			{ messages: [], tools: [tool] },
			{
				model: createModel(),
				convertToLlm: identityConverter,
				toolExecution: "sequential",
				getSteeringMessages: async () => {
					if (!steeringReady || steeringDelivered) return [];
					steeringDelivered = true;
					return [createUserMessage("late steering")];
				},
				getFollowUpMessages: async () => {
					followUpPolls++;
					if (followUpPolls !== 1) return [];
					markFollowUpStarted?.();
					await followUpGate;
					return [createUserMessage("queued follow-up")];
				},
			},
			async () => {},
			undefined,
			(_model, providerContext) => {
				const mockStream = new MockAssistantStream();
				userMessagesByCall.push(
					providerContext.messages.flatMap((message) =>
						message.role === "user" && typeof message.content === "string" ? [message.content] : [],
					),
				);
				queueMicrotask(() => {
					llmCalls++;
					mockStream.push(
						llmCalls === 1
							? {
									type: "done",
									reason: "toolUse",
									message: createAssistantMessage(
										[{ type: "toolCall", id: "tool-1", name: "terminate", arguments: {} }],
										"toolUse",
									),
								}
							: {
									type: "done",
									reason: "stop",
									message: createAssistantMessage([{ type: "text", text: `response ${llmCalls}` }]),
								},
					);
				});
				return mockStream;
			},
		);

		await followUpStarted;
		steeringReady = true;
		releaseFollowUp?.();
		const messages = await loopPromise;

		expect(executions).toBe(1);
		expect(llmCalls).toBe(3);
		expect(followUpPolls).toBe(2);
		expect(userMessagesByCall).toEqual([
			["start"],
			["start", "late steering"],
			["start", "late steering", "queued follow-up"],
		]);
		expect(
			messages.flatMap((message) =>
				message.role === "user" && typeof message.content === "string" ? [message.content] : [],
			),
		).toEqual(["start", "late steering", "queued follow-up"]);
	});

	it("should settle every sequential tool call and preserve a valid provider transcript on abort", async () => {
		const controller = new AbortController();
		let activeToolSignal: AbortSignal | undefined;
		let markActiveToolStarted: (() => void) | undefined;
		const activeToolStarted = new Promise<void>((resolve) => {
			markActiveToolStarted = resolve;
		});
		const executed: string[] = [];
		const toolSchema = Type.Object({ value: Type.String() });
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "work",
			label: "Work",
			description: "Complete or wait until aborted",
			parameters: toolSchema,
			async execute(_toolCallId, params, signal) {
				executed.push(params.value);
				if (params.value === "completed") {
					return { content: [{ type: "text", text: "completed" }], details: { value: params.value } };
				}
				activeToolSignal = signal;
				markActiveToolStarted?.();
				await new Promise<void>((_resolve, reject) => {
					signal?.addEventListener("abort", () => reject(new Error("active tool aborted")), { once: true });
				});
				throw new Error("unreachable");
			},
		};
		let llmCalls = 0;
		let providerTranscriptWasValid = false;
		const stream = agentLoop(
			[createUserMessage("run all")],
			{ messages: [], tools: [tool] },
			{
				model: createModel(),
				convertToLlm: identityConverter,
				toolExecution: "sequential",
			},
			controller.signal,
			(_model, providerContext) => {
				const mockStream = new MockAssistantStream();
				queueMicrotask(() => {
					if (llmCalls++ === 0) {
						mockStream.push({
							type: "done",
							reason: "toolUse",
							message: createAssistantMessage(
								[
									{ type: "toolCall", id: "tool-1", name: "work", arguments: { value: "completed" } },
									{ type: "toolCall", id: "tool-2", name: "work", arguments: { value: "active" } },
									{ type: "toolCall", id: "tool-3", name: "work", arguments: { value: "unstarted" } },
								],
								"toolUse",
							),
						});
					} else {
						expectCompleteToolTranscript(providerContext.messages);
						providerTranscriptWasValid = true;
						mockStream.push({ type: "error", reason: "aborted", error: createAssistantMessage([], "aborted") });
					}
				});
				return mockStream;
			},
		);

		await activeToolStarted;
		controller.abort();
		const messages = await stream.result();
		const results = messages.filter((message): message is ToolResultMessage => message.role === "toolResult");

		expect(activeToolSignal).toBe(controller.signal);
		expect(activeToolSignal?.aborted).toBe(true);
		expect(executed).toEqual(["completed", "active"]);
		expect(results.map((result) => result.toolCallId)).toEqual(["tool-1", "tool-2", "tool-3"]);
		expect(results.map((result) => result.content[0])).toEqual([
			{ type: "text", text: "completed" },
			{ type: "text", text: "active tool aborted" },
			{ type: "text", text: "Skipped because the operation was aborted before this tool call started." },
		]);
		expect(providerTranscriptWasValid).toBe(true);
		expectCompleteToolTranscript(identityConverter(messages));
	});

	it("should settle prepared and unstarted parallel calls when abort occurs during preflight", async () => {
		const controller = new AbortController();
		let markActivePreflightStarted: (() => void) | undefined;
		const activePreflightStarted = new Promise<void>((resolve) => {
			markActivePreflightStarted = resolve;
		});
		const preflights: string[] = [];
		const executed: string[] = [];
		const toolSchema = Type.Object({ value: Type.String() });
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "work",
			label: "Work",
			description: "Work tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value);
				return { content: [{ type: "text", text: params.value }], details: { value: params.value } };
			},
		};
		let llmCalls = 0;
		let providerTranscriptWasValid = false;
		const stream = agentLoop(
			[createUserMessage("run all")],
			{ messages: [], tools: [tool] },
			{
				model: createModel(),
				convertToLlm: identityConverter,
				toolExecution: "parallel",
				beforeToolCall: async ({ toolCall }, signal) => {
					preflights.push(toolCall.id);
					if (toolCall.id === "tool-2") {
						markActivePreflightStarted?.();
						await new Promise<void>((resolve) => {
							signal?.addEventListener("abort", () => resolve(), { once: true });
						});
					}
					return undefined;
				},
			},
			controller.signal,
			(_model, providerContext) => {
				const mockStream = new MockAssistantStream();
				queueMicrotask(() => {
					if (llmCalls++ === 0) {
						mockStream.push({
							type: "done",
							reason: "toolUse",
							message: createAssistantMessage(
								[
									{ type: "toolCall", id: "tool-1", name: "work", arguments: { value: "prepared" } },
									{ type: "toolCall", id: "tool-2", name: "work", arguments: { value: "active-preflight" } },
									{ type: "toolCall", id: "tool-3", name: "work", arguments: { value: "unstarted" } },
								],
								"toolUse",
							),
						});
					} else {
						expectCompleteToolTranscript(providerContext.messages);
						providerTranscriptWasValid = true;
						mockStream.push({ type: "error", reason: "aborted", error: createAssistantMessage([], "aborted") });
					}
				});
				return mockStream;
			},
		);

		await activePreflightStarted;
		controller.abort();
		const messages = await stream.result();
		const results = messages.filter((message): message is ToolResultMessage => message.role === "toolResult");

		expect(preflights).toEqual(["tool-1", "tool-2"]);
		expect(executed).toEqual([]);
		expect(results.map((result) => result.toolCallId)).toEqual(["tool-1", "tool-2", "tool-3"]);
		expect(results.map((result) => result.content[0])).toEqual([
			{ type: "text", text: "Operation aborted before tool execution started." },
			{ type: "text", text: "Operation aborted" },
			{ type: "text", text: "Skipped because the operation was aborted before this tool call started." },
		]);
		expect(providerTranscriptWasValid).toBe(true);
		expectCompleteToolTranscript(identityConverter(messages));
	});

	it("should force sequential execution when a tool has executionMode=sequential even with default parallel config", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		let firstResolved = false;
		let parallelObserved = false;
		let releaseFirst: (() => void) | undefined;
		const firstDone = new Promise<void>((resolve) => {
			releaseFirst = resolve;
		});

		const slowTool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "slow",
			label: "Slow",
			description: "Slow tool",
			parameters: toolSchema,
			executionMode: "sequential",
			async execute(_toolCallId, params) {
				if (params.value === "first") {
					await firstDone;
					firstResolved = true;
				}
				if (params.value === "second" && !firstResolved) {
					parallelObserved = true;
				}
				return {
					content: [{ type: "text", text: `slow: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			messages: [],
			tools: [slowTool],
		};

		const userPrompt: AgentMessage = createUserMessage("run both");
		// config is parallel (default), but tool forces sequential
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
		};

		let callIndex = 0;
		const stream = agentLoop([userPrompt], context, config, undefined, () => {
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				if (callIndex === 0) {
					const message = createAssistantMessage(
						[
							{ type: "toolCall", id: "tool-1", name: "slow", arguments: { value: "first" } },
							{ type: "toolCall", id: "tool-2", name: "slow", arguments: { value: "second" } },
						],
						"toolUse",
					);
					mockStream.push({ type: "done", reason: "toolUse", message });
					setTimeout(() => releaseFirst?.(), 20);
				} else {
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					mockStream.push({ type: "done", reason: "stop", message });
				}
				callIndex++;
			});
			return mockStream;
		});

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		// With sequential execution, second tool should NOT start before first finishes
		expect(parallelObserved).toBe(false);

		const toolResultIds = events.flatMap((event) => {
			if (event.type !== "message_end" || event.message.role !== "toolResult") {
				return [];
			}
			return [event.message.toolCallId];
		});
		expect(toolResultIds).toEqual(["tool-1", "tool-2"]);
	});

	it("should force sequential execution when one of multiple tools has executionMode=sequential", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const executionOrder: string[] = [];
		let releaseSlow: (() => void) | undefined;
		const slowDone = new Promise<void>((resolve) => {
			releaseSlow = resolve;
		});

		const slowTool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "slow",
			label: "Slow",
			description: "Slow tool",
			parameters: toolSchema,
			executionMode: "sequential",
			async execute(_toolCallId, params) {
				executionOrder.push(`slow:${params.value}`);
				if (params.value === "a") {
					await slowDone;
				}
				return {
					content: [{ type: "text", text: `slow: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const fastTool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "fast",
			label: "Fast",
			description: "Fast tool",
			parameters: toolSchema,
			// no executionMode = defaults to parallel
			async execute(_toolCallId, params) {
				executionOrder.push(`fast:${params.value}`);
				return {
					content: [{ type: "text", text: `fast: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			messages: [],
			tools: [slowTool, fastTool],
		};

		const userPrompt: AgentMessage = createUserMessage("run both");
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			// parallel by default, but slowTool forces sequential
		};

		let callIndex = 0;
		const stream = agentLoop([userPrompt], context, config, undefined, () => {
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				if (callIndex === 0) {
					const message = createAssistantMessage(
						[
							{ type: "toolCall", id: "tool-1", name: "slow", arguments: { value: "a" } },
							{ type: "toolCall", id: "tool-2", name: "fast", arguments: { value: "b" } },
						],
						"toolUse",
					);
					mockStream.push({ type: "done", reason: "toolUse", message });
					setTimeout(() => releaseSlow?.(), 20);
				} else {
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					mockStream.push({ type: "done", reason: "stop", message });
				}
				callIndex++;
			});
			return mockStream;
		});

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		// Fast tool should NOT run before slow tool finishes
		expect(executionOrder[0]).toBe("slow:a");
		expect(executionOrder).toContain("fast:b");
	});

	it("should allow parallel execution when all tools have executionMode=parallel", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		let firstResolved = false;
		let parallelObserved = false;
		let releaseFirst: (() => void) | undefined;
		const firstDone = new Promise<void>((resolve) => {
			releaseFirst = resolve;
		});

		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			executionMode: "parallel",
			async execute(_toolCallId, params) {
				if (params.value === "first") {
					await firstDone;
					firstResolved = true;
				}
				if (params.value === "second" && !firstResolved) {
					parallelObserved = true;
				}
				return {
					content: [{ type: "text", text: `echoed: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			messages: [],
			tools: [tool],
		};

		const userPrompt: AgentMessage = createUserMessage("echo both");
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
		};

		let callIndex = 0;
		const stream = agentLoop([userPrompt], context, config, undefined, () => {
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				if (callIndex === 0) {
					const message = createAssistantMessage(
						[
							{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "first" } },
							{ type: "toolCall", id: "tool-2", name: "echo", arguments: { value: "second" } },
						],
						"toolUse",
					);
					mockStream.push({ type: "done", reason: "toolUse", message });
					setTimeout(() => releaseFirst?.(), 20);
				} else {
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					mockStream.push({ type: "done", reason: "stop", message });
				}
				callIndex++;
			});
			return mockStream;
		});

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		// With executionMode=parallel, second tool should start before first finishes
		expect(parallelObserved).toBe(true);
	});

	it("should use prepareNextTurn snapshot before continuing", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				return {
					content: [{ type: "text", text: `echoed: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};
		const context: AgentContext = {
			messages: [],
			tools: [tool],
		};
		let convertedSecondTurnHasUpdate = false;
		let prepareCalls = 0;
		let prepared = false;
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			prepareNextTurn: async ({ context: currentContext }) => {
				prepareCalls++;
				if (prepared) return undefined;
				prepared = true;
				return {
					context: {
						messages: currentContext.messages.slice(),
						tools: currentContext.tools,
					},
					messages: [{ role: "system", content: "updated guidance", timestamp: 1 }],
				};
			},
		};

		let llmCalls = 0;
		const stream = agentLoop([createUserMessage("echo something")], context, config, undefined, (_model, ctx) => {
			llmCalls++;
			if (llmCalls === 2) {
				convertedSecondTurnHasUpdate = ctx.messages.some(
					(message) => message.role === "system" && message.content === "updated guidance",
				);
			}
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				if (llmCalls === 1) {
					mockStream.push({
						type: "done",
						reason: "toolUse",
						message: createAssistantMessage(
							[{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "hello" } }],
							"toolUse",
						),
					});
				} else {
					mockStream.push({
						type: "done",
						reason: "stop",
						message: createAssistantMessage([{ type: "text", text: "done" }]),
					});
				}
			});
			return mockStream;
		});

		for await (const _event of stream) {
			// consume
		}

		expect(llmCalls).toBe(2);
		expect(prepareCalls).toBe(1);
		expect(convertedSecondTurnHasUpdate).toBe(true);
	});

	it("should stop after the current turn when shouldStopAfterTurn returns true", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const executed: string[] = [];
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value);
				return {
					content: [{ type: "text", text: `echoed: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			messages: [],
			tools: [tool],
		};

		let steeringPolls = 0;
		let followUpPolls = 0;
		let callbackToolResultIds: string[] = [];
		let callbackContextRoles: string[] = [];
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			getSteeringMessages: async () => {
				steeringPolls++;
				return [];
			},
			getFollowUpMessages: async () => {
				followUpPolls++;
				return [createUserMessage("follow up should stay queued")];
			},
			shouldStopAfterTurn: async ({ message, toolResults, context }) => {
				expect(message.role).toBe("assistant");
				callbackToolResultIds = toolResults.map((toolResult) => toolResult.toolCallId);
				callbackContextRoles = context.messages.map((contextMessage) => contextMessage.role);
				return true;
			},
		};

		let llmCalls = 0;
		const stream = agentLoop([createUserMessage("echo something")], context, config, undefined, () => {
			llmCalls++;
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				if (llmCalls === 1) {
					const message = createAssistantMessage(
						[{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "hello" } }],
						"toolUse",
					);
					mockStream.push({ type: "done", reason: "toolUse", message });
				} else {
					mockStream.push({
						type: "done",
						reason: "stop",
						message: createAssistantMessage([{ type: "text", text: "should not run" }]),
					});
				}
			});
			return mockStream;
		});

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		const messages = await stream.result();
		expect(llmCalls).toBe(1);
		expect(executed).toEqual(["hello"]);
		// Poll at run start, before the first tool, after preflight, after turn_end,
		// and after the awaited stop callback.
		expect(steeringPolls).toBe(5);
		expect(followUpPolls).toBe(0);
		expect(callbackToolResultIds).toEqual(["tool-1"]);
		expect(callbackContextRoles).toEqual(["system", "user", "assistant", "toolResult"]);
		// The context declares no tools, so the loop announces the loadout with a system message.
		expect(messages.map((message) => message.role)).toEqual(["system", "user", "assistant", "toolResult"]);
		expect(events.map((event) => event.type)).toEqual([
			"agent_start",
			"turn_start",
			"message_start",
			"message_end",
			"message_start",
			"message_end",
			"message_start",
			"message_end",
			"tool_execution_start",
			"tool_execution_end",
			"message_start",
			"message_end",
			"turn_end",
			"agent_end",
		]);
	});

	it("should stop after a tool batch when every tool result sets terminate=true", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				return {
					content: [{ type: "text", text: `echoed: ${params.value}` }],
					details: { value: params.value },
					terminate: true,
				};
			},
		};

		const context: AgentContext = {
			messages: [],
			tools: [tool],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
		};

		let llmCalls = 0;
		const stream = agentLoop([createUserMessage("echo something")], context, config, undefined, () => {
			llmCalls++;
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				const message = createAssistantMessage(
					[{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "hello" } }],
					"toolUse",
				);
				mockStream.push({ type: "done", reason: "toolUse", message });
			});
			return mockStream;
		});

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		const messages = await stream.result();
		expect(llmCalls).toBe(1);
		expect(messages.map((message) => message.role)).toEqual(["system", "user", "assistant", "toolResult"]);
		expect(events.filter((event) => event.type === "turn_end")).toHaveLength(1);
	});

	it("should stop after a blocked tool call when beforeToolCall sets terminate=true", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		let executed = false;
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute() {
				executed = true;
				return {
					content: [{ type: "text", text: "should not execute" }],
					details: { value: "unexpected" },
				};
			},
		};
		const context: AgentContext = {
			messages: [],
			tools: [tool],
		};
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			beforeToolCall: async () => ({ block: true, reason: "Blocked by policy", terminate: true }),
		};

		let llmCalls = 0;
		const stream = agentLoop([createUserMessage("echo something")], context, config, undefined, () => {
			llmCalls++;
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				const message =
					llmCalls === 1
						? createAssistantMessage(
								[{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "hello" } }],
								"toolUse",
							)
						: createAssistantMessage([{ type: "text", text: "should not run" }]);
				mockStream.push({ type: "done", reason: llmCalls === 1 ? "toolUse" : "stop", message });
			});
			return mockStream;
		});

		for await (const _event of stream) {
			// consume
		}

		const messages = await stream.result();
		const toolResult = messages.find((message) => message.role === "toolResult");
		expect(executed).toBe(false);
		expect(llmCalls).toBe(1);
		expect(toolResult?.role === "toolResult" ? toolResult.isError : false).toBe(true);
		expect(toolResult?.role === "toolResult" ? toolResult.content : []).toContainEqual({
			type: "text",
			text: "Blocked by policy",
		});
	});

	it("should continue after a mixed batch with one terminating blocked call", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const executed: string[] = [];
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				executed.push(params.value);
				return {
					content: [{ type: "text", text: `echoed: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};
		const context: AgentContext = {
			messages: [],
			tools: [tool],
		};
		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			toolExecution: "parallel",
			beforeToolCall: async ({ args }) => {
				const { value } = args as { value: string };
				return value === "first" ? { block: true, reason: "Blocked first", terminate: true } : undefined;
			},
		};

		let llmCalls = 0;
		const stream = agentLoop([createUserMessage("echo both")], context, config, undefined, () => {
			llmCalls++;
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				const message =
					llmCalls === 1
						? createAssistantMessage(
								[
									{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "first" } },
									{ type: "toolCall", id: "tool-2", name: "echo", arguments: { value: "second" } },
								],
								"toolUse",
							)
						: createAssistantMessage([{ type: "text", text: "done" }]);
				mockStream.push({ type: "done", reason: llmCalls === 1 ? "toolUse" : "stop", message });
			});
			return mockStream;
		});

		for await (const _event of stream) {
			// consume
		}

		expect(executed).toEqual(["second"]);
		expect(llmCalls).toBe(2);
	});

	it("should continue after parallel tool calls when not all tool results terminate", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				return {
					content: [{ type: "text", text: `echoed: ${params.value}` }],
					details: { value: params.value },
					terminate: params.value === "first",
				};
			},
		};

		const context: AgentContext = {
			messages: [],
			tools: [tool],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			toolExecution: "parallel",
		};

		let callIndex = 0;
		const stream = agentLoop([createUserMessage("echo both")], context, config, undefined, () => {
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				if (callIndex === 0) {
					const message = createAssistantMessage(
						[
							{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "first" } },
							{ type: "toolCall", id: "tool-2", name: "echo", arguments: { value: "second" } },
						],
						"toolUse",
					);
					mockStream.push({ type: "done", reason: "toolUse", message });
				} else {
					const message = createAssistantMessage([{ type: "text", text: "done" }]);
					mockStream.push({ type: "done", reason: "stop", message });
				}
				callIndex++;
			});
			return mockStream;
		});

		for await (const _event of stream) {
			// consume
		}

		const messages = await stream.result();
		expect(callIndex).toBe(2);
		expect(messages.map((message) => message.role)).toEqual([
			"system",
			"user",
			"assistant",
			"toolResult",
			"toolResult",
			"assistant",
		]);
	});

	it("should allow afterToolCall to mark a tool batch as terminating", async () => {
		const toolSchema = Type.Object({ value: Type.String() });
		const tool: AgentTool<typeof toolSchema, { value: string }> = {
			name: "echo",
			label: "Echo",
			description: "Echo tool",
			parameters: toolSchema,
			async execute(_toolCallId, params) {
				return {
					content: [{ type: "text", text: `echoed: ${params.value}` }],
					details: { value: params.value },
				};
			},
		};

		const context: AgentContext = {
			messages: [],
			tools: [tool],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
			afterToolCall: async () => ({ terminate: true }),
		};

		let llmCalls = 0;
		const stream = agentLoop([createUserMessage("echo something")], context, config, undefined, () => {
			llmCalls++;
			const mockStream = new MockAssistantStream();
			queueMicrotask(() => {
				const message = createAssistantMessage(
					[{ type: "toolCall", id: "tool-1", name: "echo", arguments: { value: "hello" } }],
					"toolUse",
				);
				mockStream.push({ type: "done", reason: "toolUse", message });
			});
			return mockStream;
		});

		for await (const _event of stream) {
			// consume
		}

		expect(llmCalls).toBe(1);
	});
});

describe("agentLoopContinue with AgentMessage", () => {
	it("should throw when context has no messages", () => {
		const context: AgentContext = {
			messages: [],
			tools: [],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
		};

		expect(() =>
			agentLoopContinue(context, config, undefined, () => {
				throw new Error("Unexpected stream call");
			}),
		).toThrow("Cannot continue: no messages in context");
	});

	it("should continue from existing context without emitting user message events", async () => {
		const userMessage: AgentMessage = createUserMessage("Hello");

		const context: AgentContext = {
			messages: [userMessage],
			tools: [],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: identityConverter,
		};

		const streamFn = () => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				const message = createAssistantMessage([{ type: "text", text: "Response" }]);
				stream.push({ type: "done", reason: "stop", message });
			});
			return stream;
		};

		const events: AgentEvent[] = [];
		const stream = agentLoopContinue(context, config, undefined, streamFn);

		for await (const event of stream) {
			events.push(event);
		}

		const messages = await stream.result();

		// Should only return the new assistant message (not the existing user message)
		expect(messages.length).toBe(1);
		expect(messages[0].role).toBe("assistant");

		// Should NOT have user message events (that's the key difference from agentLoop)
		const messageEndEvents = events.filter((e) => e.type === "message_end");
		expect(messageEndEvents.length).toBe(1);
		expect((messageEndEvents[0] as any).message.role).toBe("assistant");
	});

	it("should allow custom message types as last message (caller responsibility)", async () => {
		// Custom message that will be converted to user message by convertToLlm
		interface CustomMessage {
			role: "custom";
			text: string;
			timestamp: number;
		}

		const customMessage: CustomMessage = {
			role: "custom",
			text: "Hook content",
			timestamp: Date.now(),
		};

		const context: AgentContext = {
			messages: [customMessage as unknown as AgentMessage],
			tools: [],
		};

		const config: AgentLoopConfig = {
			model: createModel(),
			convertToLlm: (messages) => {
				// Convert custom to user message
				return messages
					.map((m) => {
						if ((m as any).role === "custom") {
							return {
								role: "user" as const,
								content: (m as any).text,
								timestamp: m.timestamp,
							};
						}
						return m;
					})
					.filter((m) => m.role === "user" || m.role === "assistant" || m.role === "toolResult") as Message[];
			},
		};

		const streamFn = () => {
			const stream = new MockAssistantStream();
			queueMicrotask(() => {
				const message = createAssistantMessage([{ type: "text", text: "Response to custom message" }]);
				stream.push({ type: "done", reason: "stop", message });
			});
			return stream;
		};

		// Should not throw - the custom message will be converted to user message
		const stream = agentLoopContinue(context, config, undefined, streamFn);

		const events: AgentEvent[] = [];
		for await (const event of stream) {
			events.push(event);
		}

		const messages = await stream.result();
		expect(messages.length).toBe(1);
		expect(messages[0].role).toBe("assistant");
	});
});
