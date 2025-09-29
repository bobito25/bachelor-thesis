

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(ontable a)
(on b c)
(on c d)
(on d a)
(ontable e)
(on f e)
(clear b)
(clear f)
)
(:goal
(and
(on a f)
(on d c)
(on f d))
)
)


