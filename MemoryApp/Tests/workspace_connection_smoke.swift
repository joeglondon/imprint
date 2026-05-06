import Foundation

func expect(_ condition: @autoclosure () -> Bool, _ message: String) {
    if !condition() {
        fatalError(message)
    }
}

@main
struct WorkspaceConnectionSmoke {
    static func main() {
        let connections = WorkspaceConnection.defaults

        expect(connections.map(\.kind) == WorkspaceConnectionKind.allCases, "defaults should cover every workspace connection")
        expect(connections.allSatisfy { !$0.isRequired }, "workspace connections should be optional by default")
        expect(connections.allSatisfy { $0.status == .disconnected }, "workspace connections should start disconnected")

        let email = WorkspaceConnection(kind: .email, status: .disconnected, isRequired: false)
        expect(email.actionTitle == "Connect", "disconnected workspace connections should expose a connect action")
        expect(email.statusLabel == "Optional", "optional disconnected workspace connections should be labeled optional")
    }
}
