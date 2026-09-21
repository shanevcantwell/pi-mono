import { afterEach, describe, expect, it, vi } from "vitest";
import { InteractiveMode } from "../src/modes/interactive/interactive-mode.ts";

type InterruptContext = {
	session: {
		isStreaming: boolean;
		isBashRunning: boolean;
		abortBash: () => void;
	};
	lastSigintTime: number;
	restoreQueuedMessagesToEditor: (options: { abort: boolean }) => number;
	clearEditor: () => void;
	shutdown: () => Promise<void>;
};

type InteractiveModePrivate = {
	handleCtrlC(this: InterruptContext): void;
};

const interactiveModePrototype = InteractiveMode.prototype as unknown as InteractiveModePrivate;

function createContext(overrides: Partial<InterruptContext["session"]> = {}): InterruptContext {
	return {
		session: {
			isStreaming: false,
			isBashRunning: false,
			abortBash: vi.fn(),
			...overrides,
		},
		lastSigintTime: 0,
		restoreQueuedMessagesToEditor: vi.fn(() => 0),
		clearEditor: vi.fn(),
		shutdown: vi.fn(async () => {}),
	};
}

describe("InteractiveMode Ctrl-C interruption", () => {
	afterEach(() => {
		vi.restoreAllMocks();
	});

	it("uses the Escape abort-and-queue-restore path while an agent response or tool is active", () => {
		const context = createContext({ isStreaming: true });
		context.lastSigintTime = 900;

		interactiveModePrototype.handleCtrlC.call(context);

		expect(context.lastSigintTime).toBe(0);
		expect(context.restoreQueuedMessagesToEditor).toHaveBeenCalledWith({ abort: true });
		expect(context.clearEditor).not.toHaveBeenCalled();
		expect(context.shutdown).not.toHaveBeenCalled();
	});

	it("aborts an active user bash command", () => {
		const context = createContext({ isBashRunning: true });

		interactiveModePrototype.handleCtrlC.call(context);

		expect(context.session.abortBash).toHaveBeenCalledTimes(1);
		expect(context.clearEditor).not.toHaveBeenCalled();
	});

	it("resets an earlier idle press when active Ctrl-C is consumed as abort", () => {
		vi.spyOn(Date, "now").mockReturnValueOnce(1_000).mockReturnValueOnce(1_200);
		const context = createContext();

		interactiveModePrototype.handleCtrlC.call(context);
		expect(context.lastSigintTime).toBe(1_000);

		context.session.isStreaming = true;
		interactiveModePrototype.handleCtrlC.call(context);
		expect(context.lastSigintTime).toBe(0);

		context.session.isStreaming = false;
		interactiveModePrototype.handleCtrlC.call(context);
		expect(context.clearEditor).toHaveBeenCalledTimes(2);
		expect(context.shutdown).not.toHaveBeenCalled();
	});

	it("preserves idle clear and second-press exit behavior", () => {
		vi.spyOn(Date, "now").mockReturnValueOnce(1_000).mockReturnValueOnce(1_300);
		const context = createContext();

		interactiveModePrototype.handleCtrlC.call(context);
		expect(context.clearEditor).toHaveBeenCalledTimes(1);
		expect(context.shutdown).not.toHaveBeenCalled();

		interactiveModePrototype.handleCtrlC.call(context);
		expect(context.clearEditor).toHaveBeenCalledTimes(1);
		expect(context.shutdown).toHaveBeenCalledTimes(1);
	});
});
