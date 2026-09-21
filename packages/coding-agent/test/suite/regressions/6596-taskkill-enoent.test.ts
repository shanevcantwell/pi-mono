import type { ChildProcess } from "node:child_process";
import { EventEmitter } from "node:events";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

const { spawnMock } = vi.hoisted(() => ({ spawnMock: vi.fn() }));

vi.mock("child_process", async (importOriginal) => {
	const actual = await importOriginal<typeof import("child_process")>();
	return { ...actual, spawn: spawnMock };
});

import { terminateOwnedDetachedChild } from "../../../src/utils/shell.ts";

async function withWindowsPlatform<T>(test: () => Promise<T>): Promise<T> {
	const platformDescriptor = Object.getOwnPropertyDescriptor(process, "platform");
	try {
		Object.defineProperty(process, "platform", { configurable: true, value: "win32" });
		return await test();
	} finally {
		if (platformDescriptor) Object.defineProperty(process, "platform", platformDescriptor);
	}
}

function fakeOwnedChild(): ChildProcess {
	const child = new EventEmitter() as ChildProcess;
	Object.assign(child, {
		pid: 1234,
		exitCode: null,
		signalCode: null,
		kill: vi.fn(() => true),
	});
	return child;
}

afterEach(() => {
	vi.useRealTimers();
	spawnMock.mockReset();
});

describe("issue #6596 taskkill failures", () => {
	it.each(["synchronous spawn failure", "asynchronous spawn error", "non-zero exit"])(
		"falls back to directly killing the owned child on %s",
		async (failure) => {
			const taskkill = new EventEmitter() as ChildProcess;
			const ownedChild = fakeOwnedChild();
			const previousSystemRoot = process.env.SystemRoot;
			process.env.SystemRoot = "C:\\CustomWindows";
			if (failure === "synchronous spawn failure") {
				spawnMock.mockImplementation(() => {
					throw new Error("spawn taskkill failed");
				});
			} else {
				spawnMock.mockReturnValue(taskkill);
			}

			try {
				await withWindowsPlatform(async () => {
					const termination = terminateOwnedDetachedChild({ child: ownedChild, detachedOnUnix: true });
					if (failure === "asynchronous spawn error") taskkill.emit("error", new Error("spawn taskkill ENOENT"));
					if (failure === "non-zero exit") taskkill.emit("close", 1);
					await expect(termination).resolves.toBe(true);
				});
			} finally {
				if (previousSystemRoot === undefined) delete process.env.SystemRoot;
				else process.env.SystemRoot = previousSystemRoot;
			}

			expect(spawnMock).toHaveBeenCalledWith(
				join("C:\\CustomWindows", "System32", "taskkill.exe"),
				["/F", "/T", "/PID", "1234"],
				{ detached: true, stdio: "ignore", windowsHide: true },
			);
			expect(ownedChild.kill).toHaveBeenCalledWith("SIGKILL");
		},
	);

	it("terminates and unreferences a timed-out taskkill helper before falling back", async () => {
		vi.useFakeTimers();
		const helperKill = vi.fn(() => true);
		const helperUnref = vi.fn();
		const taskkill = Object.assign(new EventEmitter(), {
			kill: helperKill,
			unref: helperUnref,
		}) as unknown as ChildProcess;
		const ownedChild = fakeOwnedChild();
		spawnMock.mockReturnValue(taskkill);

		const termination = withWindowsPlatform(() =>
			terminateOwnedDetachedChild({ child: ownedChild, detachedOnUnix: true }),
		);
		let settled = false;
		void termination.finally(() => {
			settled = true;
		});

		await vi.advanceTimersByTimeAsync(999);
		expect(settled).toBe(false);
		await vi.advanceTimersByTimeAsync(1);
		await expect(termination).resolves.toBe(true);

		expect(helperKill).toHaveBeenCalledWith("SIGKILL");
		expect(helperUnref).toHaveBeenCalledOnce();
		expect(ownedChild.kill).toHaveBeenCalledWith("SIGKILL");
		expect(helperKill.mock.invocationCallOrder[0]).toBeLessThan(
			(ownedChild.kill as ReturnType<typeof vi.fn>).mock.invocationCallOrder[0],
		);
	});
});
